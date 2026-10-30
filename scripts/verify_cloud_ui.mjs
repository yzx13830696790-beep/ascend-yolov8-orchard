/* 通过 CDP 驱动无头浏览器，验证云服务接入后的关键交互路径。
   只做验证，不修改应用代码；失败时打印实际观察到的东西。 */
const list = await fetch('http://127.0.0.1:9222/json/list').then(r => r.json());
const page = list.find(t => t.type === 'page' && t.url.includes('orchard-yolo310b'));
if (!page) {
  console.error('找不到页面 target:', list.map(t => t.type + ' ' + t.url));
  process.exit(1);
}

const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();

ws.addEventListener('message', ev => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
  }
});
await new Promise(res => ws.addEventListener('open', res));

function send(method, params) {
  const mid = ++id;
  return new Promise(res => {
    pending.set(mid, res);
    ws.send(JSON.stringify({ id: mid, method, params }));
  });
}

async function evalJs(expr) {
  const r = await send('Runtime.evaluate', {
    expression: expr, awaitPromise: true, returnByValue: true,
  });
  if (r.result && r.result.exceptionDetails) {
    return { error: r.result.exceptionDetails.text };
  }
  return r.result && r.result.result ? r.result.result.value : null;
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// 等页面把云端状态渲染出来（说明 SDK 初始化 + 首次查询都完成了）
for (let i = 0; i < 30; i++) {
  const s = await evalJs(`document.querySelector('#cloudStatus').textContent`);
  if (s && s.length > 4) { console.log('[1] 云端状态:', s); break; }
  await sleep(500);
}

console.log('[2] 账号区:', await evalJs(`document.querySelector('#account').innerText.replace(/\\s+/g,' ')`));

// 未登录点「跑一次评测」→ 应弹出登录框，而不是发请求
await evalJs(`document.querySelector('#btnRun').click()`);
await sleep(400);
console.log('[3] 点跑评测后登录框:', await evalJs(`document.querySelector('#loginMask').className`),
            '| 提示:', await evalJs(`document.querySelector('#loginErr').textContent`));

// 手机号格式校验：10 位应当当场报错且不发请求
await evalJs(`document.querySelector('#phone').value='1380013800';
              document.querySelector('#btnSendCode').click()`);
await sleep(900);
console.log('[4] 10 位手机号提示:', await evalJs(`document.querySelector('#loginErr').textContent`));

// 合法手机号：应当走到发码请求（真实短信由平台发，这里只看有没有报网络/参数错）
await evalJs(`document.querySelector('#phone').value='13800138000';
              document.querySelector('#btnSendCode').click()`);
await sleep(3500);
console.log('[5] 合法手机号发码结果:', await evalJs(`document.querySelector('#loginErr').textContent`));

console.log('[6] 微信登录按钮存在:', await evalJs(`!!document.querySelector('#btnWxLogin')`));

// 关掉弹窗，回到看板
await evalJs(`document.querySelector('#btnCloseLogin').click();
              document.querySelector('nav button[data-tab="runs"]').click()`);
await sleep(600);
console.log('[7] 实验表:', await evalJs(`document.querySelector('#runsBody').innerText.replace(/\\s+/g,' ').slice(0,120)`));
console.log('[8] 数据来源标注:', await evalJs(`document.querySelector('#runsSource').textContent`));

ws.close();
