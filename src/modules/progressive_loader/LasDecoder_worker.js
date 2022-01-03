
import {Vector3} from "../../math/math.js";
import {Geometry} from "../../core/Geometry.js";


// from https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
function shuffle(array) {
	var currentIndex = array.length, temporaryValue, randomIndex;

	// While there remain elements to shuffle...
	while (0 !== currentIndex) {

		// Pick a remaining element...
		randomIndex = Math.floor(Math.random() * currentIndex);
		currentIndex -= 1;

		// And swap it with the current element.
		temporaryValue = array[currentIndex];
		array[currentIndex] = array[randomIndex];
		array[randomIndex] = temporaryValue;
	}

	return array;
}

function parsePoints(args){

	let {buffer, header, batchsize} = args;
	let {pointFormat, recordLength, min, max, scale} = header;

	let view = new DataView(buffer);

	let offsetRGB = {
		"0": 0,
		"1": 0,
		"2": 20,
		"3": 28,
		"4": 0,
		"5": 28,
		"6": 0,
		"7": 0,
	}[pointFormat];

	let geometry = new Geometry();
	geometry.numElements = batchsize;
	let position = new Float32Array(3 * batchsize);
	let color = new Uint8Array(4 * batchsize);
	for(let i = 0; i < batchsize; i++){

		let pointOffset = i * recordLength;
		let X = view.getInt32(pointOffset + 0, true);
		let Y = view.getInt32(pointOffset + 4, true);
		let Z = view.getInt32(pointOffset + 8, true);

		let x = X * scale.x + header.offset.x;
		let y = Y * scale.y + header.offset.y;
		let z = Z * scale.z + header.offset.z;

		position[3 * i + 0] = x;
		position[3 * i + 1] = y;
		position[3 * i + 2] = z;

		color[4 * i + 0] = view.getUint16(pointOffset + offsetRGB + 0);
		color[4 * i + 1] = view.getUint16(pointOffset + offsetRGB + 2);
		color[4 * i + 2] = view.getUint16(pointOffset + offsetRGB + 4);
		color[4 * i + 3] = 255;
	}

	let message = {
		numPoints: batchsize,
		buffers: {
			position: position,
			color: color,
		},
		min, max,
	};

	let transferables = [];
	for (let property in message.buffers) {

		let buffer = message.buffers[property];

		if(buffer instanceof ArrayBuffer){
			transferables.push(buffer);
		}else{
			transferables.push(buffer.buffer);
		}
	}

	postMessage(message, transferables);
}

async function readHeader(file){
	let buffer = await file.slice(0, 375).arrayBuffer();

	let view = new DataView(buffer);
	let versionMajor = view.getUint8(24);
	let versionMinor = view.getUint8(25);

	let numPoints = view.getUint32(107, true);
	if(versionMajor >= 1 && versionMinor >= 4){
		numPoints = Number(view.getBigInt64(247, true));
	}

	let offsetToPointData = view.getUint32(96, true);
	let recordLength = view.getUint16(105, true);
	let pointFormat = view.getUint8(104);

	let scale = new Vector3(
		view.getFloat64(131, true),
		view.getFloat64(139, true),
		view.getFloat64(147, true),
	);

	let offset = new Vector3(
		view.getFloat64(155, true),
		view.getFloat64(163, true),
		view.getFloat64(171, true),
	);

	let min = new Vector3(
		view.getFloat64(187, true),
		view.getFloat64(203, true),
		view.getFloat64(219, true),
	);

	let max = new Vector3(
		view.getFloat64(179, true),
		view.getFloat64(195, true),
		view.getFloat64(211, true),
	);

	let header = {
		versionMajor, versionMinor, 
		numPoints, pointFormat, recordLength, offsetToPointData,
		min, max, scale, offset,
	};

	return header;
}

async function loadLAS(file, header, octree_min){
	// break work down into batches
	let batchSize = 1_000_000;
	let batches = [];
	for(let i = 0; i < header.numPoints; i += batchSize){
		let batch = {
			start: i,
			count: Math.min(header.numPoints - i, batchSize),
		};

		batches.push(batch);
	}

	// process batches
	for(let batch of batches){

		let absolute_i = batch.start;

		let start = header.offsetToPointData + absolute_i * header.recordLength;
		let end = header.offsetToPointData + (absolute_i + batch.count) * header.recordLength;
		let buffer = await file.slice(start, end).arrayBuffer()
		
		parsePoints({
			buffer, header,
			batchsize: batch.count,
		});
		
	}
}

async function loadLAZ(file, header, octree_min){
	let arraybuffer = await file.arrayBuffer();

	let {Module} = await import("../../../libs/laz-perf/workers/laz-perf.js");

	// OPEN
	let instance = new Module.LASZip();
	var buf = Module._malloc(arraybuffer.byteLength);

	instance.arraybuffer = arraybuffer;
	instance.buf = buf;
	Module.HEAPU8.set(new Uint8Array(arraybuffer), buf);

	instance.open(buf, arraybuffer.byteLength);
	instance.readOffset = 0;

	console.log("opened!");

	let numExtraBytes = header.recordLength - {
		0: 20,
		1: 28,
		2: 26,
		3: 34,
		4: 57,
		5: 63,
		6: 30,
		7: 36,
	}[header.pointFormat];

	// HANDLE HEADER
	let laszipHeader = {
		pointsOffset: header.offsetToPointData,
		pointsFormatId: header.pointFormat & 0b111_111,
		pointsStructSize: header.recordLength,
		extraBytes: numExtraBytes,
		pointsCount: header.numPoints,
		scale: new Float64Array(...header.scale.toArray()),
		offset: new Float64Array(...header.offset.toArray()),
		maxs: header.max.toArray(),
		mins: header.min.toArray(),
	};
	let h = laszipHeader;
	instance.header = laszipHeader;

	console.log("headered!");


	// READ
	header.pointFormat = header.pointFormat & 0b111_111;
	let buf_read = Module._malloc(h.pointsStructSize);
	let pointsRead = 0;

	let pointsLeft = h.pointsCount;
	let maxBatchSize = 100_000;

	while(pointsLeft > 0){

		let batchsize = Math.min(pointsLeft, maxBatchSize);
		let target = new ArrayBuffer(batchsize * h.pointsStructSize);
		let target_u8 = new Uint8Array(target);

		for (let i = 0 ; i < batchsize; i++) {
			instance.getPoint(buf_read);

			let a = new Uint8Array(
					Module.HEAPU8.buffer,
					buf_read,
					h.pointsStructSize);

			target_u8.set(
					a,
					i * h.pointsStructSize,
					h.pointsStructSize);

			pointsRead++;
			pointsLeft--;
		}

		parsePoints({
			header, batchsize,
			buffer: target,
		});

	}

	

	Module._free(buf_read);


	// CLOSE 

	Module._free(instance.buf);
	instance.delete();

	


}

onmessage = async function(e){

	let {file} = e.data;

	let header = await readHeader(file);

	let compressed = (header.pointFormat & 0b11000000) > 0;

	if(compressed){
		loadLAZ(file, header);
	}else{
		loadLAS(file, header);
	}

};