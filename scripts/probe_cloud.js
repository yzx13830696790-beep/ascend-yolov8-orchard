/* 用 Node 加载官方 IIFE 构建产物，走真实 SDK 代码路径打云端数据面。
   这不是应用代码，只是验证脚本：目的是证明 SDK + publishableKey + RLS 真的通。
   浏览器全局做个最小桩即可，SDK 的数据请求只需要 fetch / storage / crypto。 */
const fs = require('fs');
const vm = require('vm');

const ENDPOINT = 'https://orchard-yolo310b.app.workbuddy.host';
const KEY = process.env.WB_PK;

// ---- 最小浏览器环境 ----
const mem = new Map();
const makeStorage = () => ({
  getItem: k => (mem.has(k) ? mem.get(k) : null),
  setItem: (k, v) => { mem.set(k, String(v)); },
  removeItem: k => { mem.delete(k); },
  clear: () => mem.clear(),
});

const g = globalThis;
g.window = g;
g.self = g;
g.sessionStorage = makeStorage();
g.localStorage = makeStorage();
Object.defineProperty(g, 'location', {
  value: {
    origin: ENDPOINT, href: ENDPOINT + '/', pathname: '/',
    search: '', hash: '', host: 'orchard-yolo310b.app.workbuddy.host',
  },
  writable: true, configurable: true,
});
g.navigator = g.navigator || { userAgent: 'node-probe' };
g.document = {
  location: g.location,
  cookie: '',
  createElement: () => ({ style: {}, setAttribute() {}, appendChild() {} }),
  addEventListener() {},
  body: { appendChild() {} },
};

vm.runInThisContext(fs.readFileSync(
  'console/static/vendor/workbuddy-cloud-sdk.js', 'utf8'));

if (!g.WorkBuddyCloud) {
  console.error('SDK 未挂到全局：构建产物或加载方式有问题');
  process.exit(1);
}

(async () => {
  const cloud = g.WorkBuddyCloud.createWorkBuddyCloud({
    endpoint: ENDPOINT,
    publishableKey: KEY,
  });

  console.log('client created:',
    typeof cloud.database.from === 'function',
    typeof cloud.auth.getSession === 'function',
    typeof cloud.storage.upload === 'function');

  const sel = await cloud.database.from('orchard_runs')
    .select('id,owner_name,name').order('created_at', { ascending: false }).limit(3);
  console.log('anon select ->', JSON.stringify(sel));

  const ins = await cloud.database.from('orchard_runs')
    .insert({ name: 'anon-probe-should-fail' }).select();
  console.log('anon insert ->', JSON.stringify(ins));

  const ses = await cloud.auth.getSession();
  console.log('session ->', JSON.stringify(ses && { data: ses.data, error: ses.error && ses.error.message }));
})();
