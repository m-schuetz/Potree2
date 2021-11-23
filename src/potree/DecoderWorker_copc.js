
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

class Stats{
	constructor(){
		name: "";
		min: null;
		max: null;
		mean: null;
	}
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
			let attributeIndex = 0;
			for(let attribute of pointAttributes.attributes){

				for(let j = 0; j < attribute.byteSize; j++){

					targetOffsets[i] = byteOffset + j;
					targetOffsetIncrements[i] = attribute.byteSize;

					i++;
				}

				byteOffset += attribute.byteSize * numPoints;
				attributeIndex++;
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

	let statsList = new Array();
	if(name === "r")
	{ // compute stats


		let outView = new DataView(outBuffer.buffer);

		let attributesByteSize = 0;
		for(let i = 0; i < pointAttributes.attributes.length; i++){
			let attribute = pointAttributes.attributes[i];
			
			let stats = new Stats();
			stats.name = attribute.name;

			if(attribute.numElements === 1){
				stats.min = Infinity;
				stats.max = -Infinity;
				stats.mean = 0;
			}else{
				stats.min = new Array(attribute.numElements).fill(Infinity);
				stats.max = new Array(attribute.numElements).fill(-Infinity);
				stats.mean = new Array(attribute.numElements).fill(0);
			}

			let readValue = null;
			let offset_to_first = numPoints * attributesByteSize;

			let reader = {
				"uint8"    : outView.getUint8.bind(outView),
				"uint16"   : outView.getUint16.bind(outView),
				"uint32"   : outView.getUint32.bind(outView),
				"int8"     : outView.getInt8.bind(outView),
				"int16"    : outView.getInt16.bind(outView),
				"int32"    : outView.getInt32.bind(outView),
				"float"    : outView.getFloat32.bind(outView),
				"double"   : outView.getFloat64.bind(outView),
			}[attribute.type.name];

			let elementByteSize = attribute.byteSize / attribute.numElements;
			if(reader){
				readValue = (index, element) => reader(offset_to_first + index * attribute.byteSize + element * elementByteSize, true);
			}

			if(attribute.name === "XYZ"){
				readValue = (index, element) => {

					let v = outView.getFloat32(offset_to_first + index * attribute.byteSize + element * 4, true);
					v = v + min[element];

					return v;
				}
			}

			if(readValue !== null){

				if(attribute.numElements === 1){
					for(let i = 0; i < numPoints; i++){

						let value = readValue(i, 0);

						stats.min = Math.min(stats.min, value);
						stats.max = Math.max(stats.max, value);
						stats.mean = stats.mean + value;
					}

					stats.mean = stats.mean / numPoints;
				}else{
					for(let i = 0; i < numPoints; i++){
						
						for(let j = 0; j < attribute.numElements; j++){
							let value = readValue(i, j);

							stats.min[j] = Math.min(stats.min[j], value);
							stats.max[j] = Math.max(stats.max[j], value);
							stats.mean[j] += value;
						}
					}

					for(let j = 0; j < attribute.numElements; j++){
						stats.mean[j] = stats.mean[j] / numPoints;
					}
				}

				
			}

			statsList.push(stats);
			attributesByteSize += attribute.byteSize;
		}

		console.log(statsList);
	}


	{
		let millies = performance.now() - tStart;
		let seconds = millies / 1000;

		let pointsPerSec = numPoints / seconds;
		let strPointsPerSec = (pointsPerSec / 1_000_000).toFixed(2);

		// console.log(`read ${numPoints.toLocaleString()} points in ${millies.toFixed(1)}ms. (${strPointsPerSec} million points / s`);
	}

	return {
		buffer: outBuffer, statsList
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
