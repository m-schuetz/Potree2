
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

	// let duration = performance.now() - tStart;
	// let pointsPerSecond = (1000 * numPoints / duration) / 1_000_000;
	// console.log(`[${name}] duration: ${duration.toFixed(1)}ms, #points: ${numPoints}, points/s: ${pointsPerSecond.toFixed(1)}M`);

	return {
		buffer: targetBuffer, 
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
