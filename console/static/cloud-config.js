/* 云服务公开配置 —— 来自 workbuddy_cloud_service 返回的 publicConfig。
 *
 * 这三个值本来就会随页面下发到浏览器（谁都能看到），安全边界不在"藏起来"，
 * 而在服务端：数据面按 Origin 精确匹配 + RLS 按 auth.uid() 隔离。
 * 任何长期密钥（服务端 key、数据库口令）都不在这里，也不应该出现在前端。
 */
window.ORCHARD_CLOUD = {
  endpoint: "https://orchard-yolo310b.app.workbuddy.host",
  oauthRelayBaseUrl: "https://www.workbuddy.cn/v2/as/genie-baas/oauth",
  publishableKey: "wbpk_mCjik8RHKDOF2uFq12mX50_njlIEM459oCgPTrDezCl4GYgqmMA7li5"
};
