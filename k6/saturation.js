import http from 'k6/http';
import { check } from 'k6';
import { Trend } from 'k6/metrics';

// A ramping arrival rate, not ramping VUs. This is the distinction that
// makes the test meaningful: VUs measure "N clients each waiting for a
// reply", which self-limits — as the server slows, the clients send less,
// and the server never actually saturates. Arrival rate holds OFFERED LOAD
// constant regardless of how the server is coping, which is what real
// traffic does and the only way to find a knee.
export const options = {
  scenarios: {
    saturation: {
      executor: 'ramping-arrival-rate',
      startRate: 10,
      timeUnit: '1s',
      preAllocatedVUs: 100,
      maxVUs: 500,
      stages: [
        { target: 10,  duration: '30s' },
        { target: 25,  duration: '30s' },
        { target: 50,  duration: '60s' },   // the spec's target
        { target: 100, duration: '30s' },
        { target: 200, duration: '30s' },   // past the knee, deliberately
        { target: 400, duration: '30s' },
      ],
    },
  },
  thresholds: {
    // Scoped to the 50 QPS stage. A global threshold would fail because of
    // the 400 QPS stage, which is SUPPOSED to fail — that stage exists to
    // find the limit, not to pass.
    'http_req_duration{stage:target}': ['p(95)<200'],
  },
};

const QUERIES = [
  'What is the grace period after F-1 program completion?',
  'What are the STEM OPT extension requirements?',
  'How does a student transfer between schools?',
  'What is the maximum period of authorized practical training?',
  'When must a student report a change of address?',
];

const DATES = ['2016-06-01', '2019-06-01', '2022-06-01', '2026-08-11'];

const selectivityTrend = new Trend('selectivity');

export default function () {
  const body = JSON.stringify({
    query: QUERIES[Math.floor(Math.random() * QUERIES.length)],
    as_of: DATES[Math.floor(Math.random() * DATES.length)],
    k: 10,
  });

  const res = http.post('http://127.0.0.1:8000/search', body, {
    headers: { 'Content-Type': 'application/json' },
  });

  check(res, {
    'status 200': (r) => r.status === 200,
    'has hits': (r) => {
      try { return JSON.parse(r.body).hits.length > 0; } catch { return false; }
    },
  });

  if (res.status === 200) {
    try { selectivityTrend.add(JSON.parse(res.body).selectivity); } catch {}
  }
}
