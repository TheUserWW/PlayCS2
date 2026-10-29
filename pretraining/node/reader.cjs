// File-based JSON protocol keeps large parser results out of shell arguments.
const fs = require('node:fs');
const parser = require('@laihoe/demoparser2');
const request = JSON.parse(fs.readFileSync(process.argv[2], 'utf8').replace(/^\uFEFF/, ''));
let result;
switch (request.operation) {
  case 'header': result = {header: parser.parseHeader(request.demo), players: parser.parsePlayerInfo(request.demo)}; break;
  case 'events': result = parser.parseEvents(request.demo, request.events, request.player_extra || [], request.other_extra || []); break;
  case 'fields': result = parser.listUpdatedFields(request.demo); break;
  case 'ticks': result = parser.parseTicks(request.demo, request.fields, request.ticks || null, request.players || null, false, false); break;
  default: throw new Error(`Unknown operation: ${request.operation}`);
}
fs.writeFileSync(request.output, JSON.stringify(result, (_, v) => typeof v === 'bigint' ? v.toString() : v));
