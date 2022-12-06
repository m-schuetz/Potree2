
class Stats{
	constructor(){
		this.name = "";
		this.min = null;
		this.max = null;
		this.mean = null;
	}
};

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

	let {name, pointAttributes, numPoints, scale, offset, min} = event.data;

	let buffer;
	if(event.data.byteSize === 0){
		buffer = new ArrayBuffer(0);
		console.warn(`loaded node with 0 bytes: ${name}`);
	}else{
		let {url, byteOffset, byteSize} = event.data;
		let first = byteOffset;
		let last = byteOffset + byteSize - 1;

		let response = await fetch(url, {
			headers: {
				'content-type': 'multipart/byteranges',
				'Range': `bytes=${first}-${last}`,
			},
		});

		buffer = await response.arrayBuffer();
	}


	let tStart = performance.now();

	buffer = new Uint8Array(buffer);
	let view = new DataView(buffer.buffer);

	// pad to multiple of 4 bytes due to GPU requirements.
	let alignedSize = buffer.byteLength + (4 - (buffer.byteLength % 4));
	let targetBuffer = new Uint8Array(alignedSize);
	let targetView = new DataView(targetBuffer.buffer);

	let byteOffset = 0;
	for (let pointAttribute of pointAttributes.attributes) {
		
		if(["POSITION_CARTESIAN", "position"].includes(pointAttribute.name)){

			for (let j = 0; j < numPoints; j++) {
				let pointOffset = j * pointAttributes.byteSize;

				let X = view.getInt32(pointOffset + byteOffset + 0, true);
				let Z = view.getInt32(pointOffset + byteOffset + 4, true);
				let Y = view.getInt32(pointOffset + byteOffset + 8, true);

				let x = parseInt(X) * scale[0] + offset[0] - min[0];
				let y = parseInt(Y) * scale[1] + offset[1] - min[1];
				let z = parseInt(Z) * scale[2] + offset[2] - min[2];

				let targetOffset = numPoints * byteOffset + j * 12;
				targetView.setFloat32(targetOffset + 0, x, true);
				targetView.setFloat32(targetOffset + 4, z, true);
				targetView.setFloat32(targetOffset + 8, y, true);
			}
		}else{

			for (let j = 0; j < numPoints; j++) {

				for(let k = 0; k < pointAttribute.byteSize; k++){
					let sourceOffset = j * pointAttributes.byteSize + byteOffset + k;
					let targetOffset = numPoints * byteOffset + j * pointAttribute.byteSize + k;

					targetBuffer[targetOffset] = buffer[sourceOffset];
				}

			}

		}
		
		byteOffset += pointAttribute.byteSize;

	}

	let statsList = new Array();
	if(name === "r")
	{ // compute stats

		let outView = new DataView(targetBuffer.buffer);

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

			if(["XYZ", "position"].includes(attribute.name)){
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

	// let duration = performance.now() - tStart;
	// let pointsPerSecond = (1000 * numPoints / duration) / 1_000_000;
	// console.log(`[${name}] duration: ${duration.toFixed(1)}ms, #points: ${numPoints}, points/s: ${pointsPerSecond.toFixed(1)}M`);

	return {
		buffer: targetBuffer, statsList
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
