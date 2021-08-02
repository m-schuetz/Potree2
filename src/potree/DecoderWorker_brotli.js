
import {BrotliDecode} from "../../libs/brotli/decode.js";

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

// Potree = {};

function dealign24b(mortoncode){
	// see https://stackoverflow.com/questions/45694690/how-i-can-remove-all-odds-bits-in-c

	// input alignment of desired bits
	// ..a..b..c..d..e..f..g..h..i..j..k..l..m..n..o..p
	let x = mortoncode;

	//          ..a..b..c..d..e..f..g..h..i..j..k..l..m..n..o..p                     ..a..b..c..d..e..f..g..h..i..j..k..l..m..n..o..p 
	//          ..a.....c.....e.....g.....i.....k.....m.....o...                     .....b.....d.....f.....h.....j.....l.....n.....p 
	//          ....a.....c.....e.....g.....i.....k.....m.....o.                     .....b.....d.....f.....h.....j.....l.....n.....p 
	x = ((x & 0b001000001000001000001000) >>  2) | ((x & 0b000001000001000001000001) >> 0);
	//          ....ab....cd....ef....gh....ij....kl....mn....op                     ....ab....cd....ef....gh....ij....kl....mn....op
	//          ....ab..........ef..........ij..........mn......                     ..........cd..........gh..........kl..........op
	//          ........ab..........ef..........ij..........mn..                     ..........cd..........gh..........kl..........op
	x = ((x & 0b000011000000000011000000) >>  4) | ((x & 0b000000000011000000000011) >> 0);
	//          ........abcd........efgh........ijkl........mnop                     ........abcd........efgh........ijkl........mnop
	//          ........abcd....................ijkl............                     ....................efgh....................mnop
	//          ................abcd....................ijkl....                     ....................efgh....................mnop
	x = ((x & 0b000000001111000000000000) >>  8) | ((x & 0b000000000000000000001111) >> 0);
	//          ................abcdefgh................ijklmnop                     ................abcdefgh................ijklmnop
	//          ................abcdefgh........................                     ........................................ijklmnop
	//          ................................abcdefgh........                     ........................................ijklmnop
	x = ((x & 0b000000000000000000000000) >> 16) | ((x & 0b000000000000000011111111) >> 0);

	// sucessfully realigned! 
	//................................abcdefghijklmnop

	return x;
}

async function load(event){

	let {name, pointAttributes, numPoints, scale, offset, min, nodeMin, nodeMax} = event.data;

	let buffer;
	if(event.data.byteSize === 0n){
		buffer = new ArrayBuffer(0);
		console.warn(`loaded node with 0 bytes: ${name}`);
	}else{
		let {url, byteOffset, byteSize} = event.data;
		let first = byteOffset;
		let last = byteOffset + byteSize - 1n;

		let response = await fetch(url, {
			headers: {
				'content-type': 'multipart/byteranges',
				'Range': `bytes=${first}-${last}`,
			},
		});

		buffer = await response.arrayBuffer();
	}

	let tStart = performance.now();

	let decoded = BrotliDecode(new Int8Array(buffer));
	let view = new DataView(decoded.buffer);

	let byteOffset = 0;
	for (let pointAttribute of pointAttributes.attributes) {
		
		if(["POSITION_CARTESIAN", "position"].includes(pointAttribute.name)){

			for (let j = 0; j < numPoints; j++) {

				let src_index = j;
				let pointOffset = byteOffset + src_index * 16;

				let mc_0 = view.getUint32(pointOffset +  4, true);
				let mc_1 = view.getUint32(pointOffset +  0, true);
				let mc_2 = view.getUint32(pointOffset + 12, true);
				let mc_3 = view.getUint32(pointOffset +  8, true);

				let X = dealign24b((mc_3 & 0x00FFFFFF) >>> 0) 
						| (dealign24b(((mc_3 >>> 24) | (mc_2 << 8)) >>> 0) << 8);

				let Y = dealign24b((mc_3 & 0x00FFFFFF) >>> 1) 
						| (dealign24b(((mc_3 >>> 24) | (mc_2 << 8)) >>> 1) << 8)
						
				let Z = dealign24b((mc_3 & 0x00FFFFFF) >>> 2) 
						| (dealign24b(((mc_3 >>> 24) | (mc_2 << 8)) >>> 2) << 8)
						
				if(mc_1 != 0 || mc_2 != 0){
					X = X | (dealign24b((mc_1 & 0x00FFFFFF) >>> 0) << 16)
						| (dealign24b(((mc_1 >>> 24) | (mc_0 << 8)) >>> 0) << 24);

					Y = Y | (dealign24b((mc_1 & 0x00FFFFFF) >>> 1) << 16)
						| (dealign24b(((mc_1 >>> 24) | (mc_0 << 8)) >>> 1) << 24);

					Z = Z | (dealign24b((mc_1 & 0x00FFFFFF) >>> 2) << 16)
						| (dealign24b(((mc_1 >>> 24) | (mc_0 << 8)) >>> 2) << 24);
				}

				let x = parseInt(X) * scale[0] + offset[0] - min[0];
				let y = parseInt(Y) * scale[1] + offset[1] - min[1];
				let z = parseInt(Z) * scale[2] + offset[2] - min[2];

				view.setFloat32(byteOffset + 16 * j + 0, x, true);
				view.setFloat32(byteOffset + 16 * j + 4, y, true);
				view.setFloat32(byteOffset + 16 * j + 8, z, true);

				// {
				// 	let cubeSize = nodeMax[0] - nodeMin[0];
				// 	// let iX = 1024 * (parseInt(X) * scale[0] + offset[0] - nodeMin[0]) / cubeSize;
				// 	// let iY = 1024 * (parseInt(Y) * scale[1] + offset[1] - nodeMin[1]) / cubeSize;
				// 	// let iZ = 1024 * (parseInt(Z) * scale[2] + offset[2] - nodeMin[2]) / cubeSize;

				// 	let iX = 1024 * (x - nodeMin[0]) / cubeSize;
				// 	let iY = 1024 * (y - nodeMin[1]) / cubeSize;
				// 	let iZ = 1024 * (z - nodeMin[2]) / cubeSize;

				// 	let clamp = (value, min, max) => Math.max(Math.min(value, max), min);

				// 	iX = Math.floor(clamp(iX, 0, 1023));
				// 	iY = Math.floor(clamp(iY, 0, 1023));
				// 	iZ = Math.floor(clamp(iZ, 0, 1023));

				// 	let encoded = (iX << 20) | (iY << 10) | iZ;

				// 	// view.setUint32(byteOffset + 16 * j + 12, encoded, true);
				// 	view.setUint32(byteOffset + 4 * j + 0, encoded, true);
				// }



			}

			byteOffset += 16 * numPoints;
		}else if(["RGBA", "rgba"].includes(pointAttribute.name)){

			for (let j = 0; j < numPoints; j++) {
				let src_index = j;
				let pointOffset = byteOffset + src_index * 8;

				let mc_0 = view.getUint32(pointOffset +  4, true);
				let mc_1 = view.getUint32(pointOffset +  0, true);

				let r = dealign24b((mc_1 & 0x00FFFFFF) >>> 0) 
						| (dealign24b(((mc_1 >>> 24) | (mc_0 << 8)) >>> 0) << 8);

				let g = dealign24b((mc_1 & 0x00FFFFFF) >>> 1) 
						| (dealign24b(((mc_1 >>> 24) | (mc_0 << 8)) >>> 1) << 8);

				let b = dealign24b((mc_1 & 0x00FFFFFF) >>> 2) 
						| (dealign24b(((mc_1 >>> 24) | (mc_0 << 8)) >>> 2) << 8);


				r = r | 255 ? r / 256 : r;
				g = g | 255 ? g / 256 : g;
				b = b | 255 ? b / 256 : b;

				view.setUint8(byteOffset + 4 * j + 0, r, true);
				view.setUint8(byteOffset + 4 * j + 1, g, true);
				view.setUint8(byteOffset + 4 * j + 2, b, true);
				view.setUint8(byteOffset + 4 * j + 3, 255, true);
			}

			byteOffset += 4 * numPoints;
		}else{
			byteOffset += numPoints * pointAttribute.byteSize;
		}

	}

	// let duration = performance.now() - tStart;
	// let pointsPerSecond = (1000 * numPoints / duration) / 1_000_000;
	// console.log(`[${name}] duration: ${duration.toFixed(1)}ms, #points: ${numPoints}, points/s: ${pointsPerSecond.toFixed(1)}M`);

	// pad to multiple of 4 bytes due to GPU requirements.
	let alignedBuffer;
	if((decoded.byteLength % 4) === 0){
		alignedBuffer = decoded;
	}else{
		let alignedSize = decoded.byteLength + (4 - (decoded.byteLength % 4));
		alignedBuffer = new Uint8Array(alignedSize);
		alignedBuffer.set(decoded);
	}

	return {
		buffer: alignedBuffer, 
		// attributeBuffers
	};
}

onmessage = async function (event) {

	try{
		let loaded = await load(event);

		let message = loaded;
		
		let transferables = [];
		// for (let property in message.attributeBuffers) {

		// 	let buffer = message.attributeBuffers[property].buffer;

		// 	if(buffer instanceof ArrayBuffer){
		// 		transferables.push(buffer);
		// 	}else{
		// 		transferables.push(buffer.buffer);
		// 	}
		// }

		transferables.push(loaded.buffer.buffer);

		postMessage(message, transferables);
	}catch(e){
		console.log(e);
		postMessage("failed");
	}

	
};
