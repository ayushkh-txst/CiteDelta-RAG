import http from 'k6/http';
import { check } from 'k6';
import { Trend } from 'k6/metrics';

// Fixed offered load for one row of the saturation table. The ramping script
// (saturation.js) is the headline test; this parametrised sibling produces
// one clean per-stage number per invocation so the table has real cells.
const RATE = Number(__ENV.RATE || 50);
const DURATION = __ENV.DURATION || '45s';

export const options = {
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
  scenarios: {
    sat: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: DURATION,
      preAllocatedVUs: 100,
      maxVUs: 400,
    },
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
  });
}
