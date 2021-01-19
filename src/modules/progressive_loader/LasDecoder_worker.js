
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

onmessage = async function(e){

	console.log("test");

	let {file, octree_min} = e.data;

	let buffer = await file.slice(0, 375).arrayBuffer()

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


	let batchSize = 1_000_000;
	let batches = [];
	for(let i = 0; i < numPoints; i += batchSize){
		let batch = {
			start: i,
			count: Math.min(numPoints - i, batchSize),
		};

		batches.push(batch);
	}
	// shuffle(batches);

	// for(let absolute_i = 0; absolute_i < numPoints; absolute_i += batchSize){
	// 	let n = Math.min(numPoints - absolute_i, batchSize);
	for(let batch of batches){

		let absolute_i = batch.start;
		let n = batch.count;

		let buffer = await file.slice(
			offsetToPointData + absolute_i * recordLength, 
			offsetToPointData + (absolute_i + n) * recordLength,
		).arrayBuffer()
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
		geometry.numElements = n;
		let position = new Float32Array(3 * n);
		let color = new Uint8Array(4 * n);
		for(let i = 0; i < n; i++){

			let pointOffset = i * recordLength;
			let X = view.getInt32(pointOffset + 0, true);
			let Y = view.getInt32(pointOffset + 4, true);
			let Z = view.getInt32(pointOffset + 8, true);

			let x = X * header.scale.x + header.offset.x - octree_min.x;
			let y = Y * header.scale.y + header.offset.y - octree_min.y;
			let z = Z * header.scale.z + header.offset.z - octree_min.z;

			position[3 * i + 0] = x;
			position[3 * i + 1] = y;
			position[3 * i + 2] = z;

			color[4 * i + 0] = view.getUint16(pointOffset + offsetRGB + 0);
			color[4 * i + 1] = view.getUint16(pointOffset + offsetRGB + 2);
			color[4 * i + 2] = view.getUint16(pointOffset + offsetRGB + 4);
			color[4 * i + 3] = 255;
		}

		let message = {
			numPoints: n,
			buffers: {
				position: position,
				color: color,
			},
			min, max,
			header,
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

	// postMessage({header});

};