
import {createLazPerf} from "../../libs/laz-perf/laz-perf.js";

let lazperf = null;

const typedArrayMapping = {
	"int8":   Int8Array,
	"int16":  Int16Array,
	"int32":  Int32Array,
	"int64":  Float64Array,
	"uint8":  Uint8Array,
	"uint16": Uint16Array,
	"uint32": Uint32Array,
	"uint64": Float64Array,
	"float":  Float32Array,
	"double": Float64Array,
};


async function load(event){

	if(lazperf === null){
		lazperf = await createLazPerf();
	}

	let {name, pointAttributes, numPoints, scale, offset, min} = event.data;
	let {pointFormat, pointRecordLength} = event.data;

	let tStart = performance.now();

	let blobPointer = null;
	let dataPointer = null;
	let decoder = null;
	let outBuffer = null;

	try{
		let {url, byteOffset, byteSize} = event.data;
		let first = byteOffset;
		let last = byteOffset + byteSize - 1;
		let response = await fetch(url, {
			headers: {
				'content-type': 'multipart/byteranges',
				'Range': `bytes=${first}-${last}`,
			},
		});

		let buffer = await response.arrayBuffer();
		let compressed = new Uint8Array(buffer);

		let pointDataRecordFormat = pointFormat;
		let pointDataRecordLength = pointRecordLength;
		let outSize = pointRecordLength * numPoints;
		let alignedOutSize = outSize + (4 - (outSize % 4));
		outBuffer = new Uint8Array(alignedOutSize);
		let outView = new DataView(outBuffer.buffer);

		blobPointer = lazperf._malloc(compressed.byteLength);
		dataPointer = lazperf._malloc(pointDataRecordLength);
		decoder = new lazperf.ChunkDecoder();

		lazperf.HEAPU8.set(
			new Uint8Array(
				compressed.buffer,
				compressed.byteOffset,
				compressed.byteLength
			),
			blobPointer
		);

		decoder.open(pointDataRecordFormat, pointDataRecordLength, blobPointer);

		let pointView = new DataView(lazperf.HEAPU8.buffer, dataPointer, pointDataRecordLength);
		let pointU8 = new Uint8Array(lazperf.HEAPU8.buffer, dataPointer, pointDataRecordLength);

		let targetOffsets = new Array(pointDataRecordLength).fill(0);
		let targetOffsetIncrements = new Array(pointDataRecordLength).fill(0);
		{
			let i = 0;
			let byteOffset = 0;
			for(let attribute of pointAttributes.attributes){

				for(let j = 0; j < attribute.byteSize; j++){

					targetOffsets[i] = byteOffset + j;
					targetOffsetIncrements[i] = attribute.byteSize;

					i++;
				}

				byteOffset += attribute.byteSize * numPoints;
			}

		}

		for (let i = 0; i < numPoints; ++i) {
			decoder.getPoint(dataPointer);

			{ // Special handling/decoding for XYZ data
				let X = pointView.getInt32(0, true);
				let Y = pointView.getInt32(4, true);
				let Z = pointView.getInt32(8, true);

				let x = X * scale[0] + offset[0] - min[0];
				let y = Y * scale[1] + offset[1] - min[1];
				let z = Z * scale[2] + offset[2] - min[2];

				outView.setFloat32(12 * i + 0, x, true);
				outView.setFloat32(12 * i + 4, y, true);
				outView.setFloat32(12 * i + 8, z, true);
			}

			// everything else: just rearrange bytes into struct-of-arrays layout
			for(let j = 12; j < pointRecordLength; j++){
				outBuffer[targetOffsets[j]] = pointU8[j];
				targetOffsets[j] += targetOffsetIncrements[j];
			}

		}
	}finally{
		if(blobPointer){
			lazperf._free(blobPointer);
			lazperf._free(dataPointer);
			decoder.delete();
		}
	}

	{
		let millies = performance.now() - tStart;
		let seconds = millies / 1000;

		let pointsPerSec = numPoints / seconds;
		let strPointsPerSec = (pointsPerSec / 1_000_000).toFixed(2);

		// console.log(`read ${numPoints.toLocaleString()} points in ${millies.toFixed(1)}ms. (${strPointsPerSec} million points / s`);
	}

	return {
		buffer: outBuffer, 
	};
}

onmessage = async function (event) {

	try{
		let loaded = await load(event);

		let message = loaded;
		
		let transferables = [];

		transferables.push(loaded.buffer.buffer);

		postMessage(message, transferables);
	}catch(e){
		console.log(e);
		postMessage("failed");
	}

	
};
