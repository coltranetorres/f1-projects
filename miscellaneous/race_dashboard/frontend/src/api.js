async function get(path) {
  const r = await fetch(`/api${path}`);
  if (!r.ok) throw new Error(`${path} failed: ${r.status}`);
  return r.json();
}

export const getRace = () => get("/race");
export const getFrames = (t0, t1) => get(`/frames?t0=${t0}&t1=${t1}`);
export const getInsights = () => get("/insights");
