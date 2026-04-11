#!/usr/bin/env node

"use strict";

const readline = require("readline");
const path = require("path");

const katexModulePath = process.argv[2];
if (!katexModulePath) {
  process.stderr.write("missing katex module path\n");
  process.exit(1);
}

const katex = require(path.resolve(katexModulePath));
const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

rl.on("line", (line) => {
  let request;
  try {
    request = JSON.parse(line);
    const html = katex.renderToString(request.expr, request.options || {});
    process.stdout.write(`${JSON.stringify({ id: request.id, html })}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const requestId = request && typeof request.id !== "undefined" ? request.id : null;
    process.stdout.write(`${JSON.stringify({ id: requestId, error: message })}\n`);
  }
});
