const http = require('http');
const { trace, SpanStatusCode } = require('@opentelemetry/api');
const tracer = trace.getTracer('tracemind-release-lab');

function quote(input) {
  const currency = input.currency || 'USD';
  return currency.toUpperCase();
}

http.createServer((request, response) => {
  if (request.url === '/health') {
    response.end('ok');
    return;
  }
  let body = '';
  request.on('data', chunk => { body += chunk; });
  request.on('end', () => {
    tracer.startActiveSpan('quote', span => {
      const context = span.spanContext();
      span.setAttribute('code.file.path', __filename);
      span.setAttribute('code.function.name', 'quote');
      span.setAttribute('code.line.number', 7);
      try {
        const currency = quote(JSON.parse(body));
        span.setStatus({ code: SpanStatusCode.OK });
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ currency, trace_id: context.traceId, span_id: context.spanId }));
      } catch (error) {
        span.recordException(error);
        span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
        response.writeHead(500, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ error: error.message, trace_id: context.traceId, span_id: context.spanId }));
      } finally {
        span.end();
      }
    });
  });
}).listen(8080, '0.0.0.0');
