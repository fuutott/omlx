// Real Alpine effects; fake time/DOM only. No npm dependencies or browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'omlx/admin/templates/chat.html'), 'utf8');
const microtasks = [], timers = new Map();
let now = 0, seq = 0;
const sandbox = {
    window: {Element: class {}}, MutationObserver: class {},
    document: {createElement: () => ({})}, console,
    queueMicrotask: fn => microtasks.push(fn),
    setTimeout: (fn, delay) => {timers.set(++seq, {fn, at: now + delay}); return seq;},
    clearTimeout: id => timers.delete(id), clearInterval: () => {},
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(root, 'omlx/admin/static/js/alpine.min.js'), 'utf8'), sandbox);
microtasks.length = 0; // Suppress DOM startup, retaining the real reactivity engine.
sandbox.Alpine = sandbox.window.Alpine;
function method(start, end) {
    const text = html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start)));
    return vm.runInContext('({' + text + '})', sandbox);
}
sandbox.methods = {
    ...method('    updateStreamingDOM(_contentTick) {', '    _updateStreamingDOMImpl() {'),
    ...method('            resetStreamSession(stream,', '            stopAllStreams() {'),
};
vm.runInContext(`
    var paints = [];
    var app = Alpine.reactive({
        ...methods,
        stream: {isStreaming:true, streamingContent:'first'},
        currentStream() {return this.stream},
        createThinkingState() {return {}},
        _updateStreamingDOMImpl() {paints.push(this.stream.streamingContent)},
    });
    var effect = Alpine.effect(() => app.currentStream()?.isStreaming &&
        app.updateStreamingDOM(app.currentStream()?.streamingContent));
`, sandbox);
function flush() {
    let budget = 100;
    while (microtasks.length && budget--) microtasks.shift()();
    assert.ok(budget > 0, 'reactive microtask loop');
}
function advance(ms) {
    const end = now + ms;
    while (true) {
        const next = [...timers].sort((a,b) => a[1].at - b[1].at)[0];
        if (!next || next[1].at > end) break;
        now = next[1].at; timers.delete(next[0]); next[1].fn(); flush();
    }
    now = end;
}
function run(code) {vm.runInContext(code, sandbox); flush();}
assert.equal(timers.size, 1);
advance(199);
assert.equal(sandbox.paints.length, 0);
run("app.stream.streamingContent='latest before first paint'");
assert.equal(timers.size, 1, 'new tokens must not postpone/duplicate pending paint');
advance(1);
assert.equal(sandbox.paints.at(-1), 'latest before first paint');
advance(2000);
assert.equal(sandbox.paints.length, 1, 'unchanged content must not redraw');
assert.equal(timers.size, 0);
// Continuous arrivals: at most five intermediate paints per second.
const before = sandbox.paints.length;
for (let i=0;i<100;i++) {run(`app.stream.streamingContent='token ${i}'`); advance(10);}
assert.equal(sandbox.paints.length - before, 5);
// Finish/cancellation cancels the pending timer, without stale paints.
run("app.stream.streamingContent='final token'");
assert.equal(timers.size, 1);
run("app.resetStreamSession(app.stream, {preserveFinalContent:true})");
assert.equal(timers.size, 0);
const finished = sandbox.paints.length;
advance(1000);
assert.equal(sandbox.paints.length, finished);
// Switching chats leaves the previous callback harmless; the new one paints.
run("app.stream={isStreaming:true, streamingContent:'old chat'}");
run("app.stream={isStreaming:true, streamingContent:'new chat'}");
advance(200);
assert.equal(sandbox.paints.length, finished + 1);
assert.equal(sandbox.paints.at(-1), 'new chat');
run("app.stream.isStreaming=false");
sandbox.Alpine.release(sandbox.effect);
console.log('PASS: no idle redraws; 5 Hz cap; coalescing; cleanup; chat switching');
