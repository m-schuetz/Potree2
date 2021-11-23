
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

		// console.log((Number(byteSize) / 1024).toFixed(1) + " kb");

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

	let outByteSize = pointAttributes.byteSize * numPoints;
	let alignedOutByteSize = outByteSize + (4 - (outByteSize % 4));
	let outBuffer = new ArrayBuffer(alignedOutByteSize);
	let outView = new DataView(outBuffer);

	let sourceByteOffset = 0;
	let targetByteOffset = 0;
	for (let pointAttribute of pointAttributes.attributes) {
		
		if(["POSITION_CARTESIAN", "position"].includes(pointAttribute.name)){

			for (let j = 0; j < numPoints; j++) {

				let src_index = j;
				let pointOffset = sourceByteOffset + src_index * 16;

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

				outView.setFloat32(targetByteOffset + 12 * j + 0, x, true);
				outView.setFloat32(targetByteOffset + 12 * j + 4, y, true);
				outView.setFloat32(targetByteOffset + 12 * j + 8, z, true);
			}

			sourceByteOffset += 16 * numPoints;
			targetByteOffset += 12 * numPoints;
		}else if(["RGBA", "rgba"].includes(pointAttribute.name)){

			for (let j = 0; j < numPoints; j++) {
				let src_index = j;
				let pointOffset = sourceByteOffset + src_index * 8;

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

				outView.setUint16(targetByteOffset + 6 * j + 0, r, true);
				outView.setUint16(targetByteOffset + 6 * j + 2, g, true);
				outView.setUint16(targetByteOffset + 6 * j + 4, b, true);
				// outView.setUint8(targetByteOffset + 4 * j + 3, 255, true);
			}

			sourceByteOffset += 4 * numPoints;
			targetByteOffset += 6 * numPoints;
		}else{
			sourceByteOffset += numPoints * pointAttribute.byteSize;
			targetByteOffset += numPoints * pointAttribute.byteSize;
		}

	}

	// let duration = performance.now() - tStart;
	// let pointsPerSecond = (1000 * numPoints / duration) / 1_000_000;
	// console.log(`[${name}] duration: ${duration.toFixed(1)}ms, #points: ${numPoints}, points/s: ${pointsPerSecond.toFixed(1)}M`);

	// {
	// 	let millies = performance.now() - tStart;
	// 	let seconds = millies / 1000;

	// 	let pointsPerSec = numPoints / seconds;
	// 	let strPointsPerSec = (pointsPerSec / 1_000_000).toFixed(2);

	// 	console.log(`read ${numPoints.toLocaleString()} points in ${millies.toFixed(1)}ms. (${strPointsPerSec} million points / s`);
	// }

	// pad to multiple of 4 bytes due to GPU requirements.
	// let alignedBuffer;
	// if((decoded.byteLength % 4) === 0){
	// 	alignedBuffer = decoded;
	// }else{
	// 	let alignedSize = decoded.byteLength + (4 - (decoded.byteLength % 4));
	// 	alignedBuffer = new Uint8Array(alignedSize);
	// 	alignedBuffer.set(decoded);
	// }

	return {
		buffer: new Uint8Array(outBuffer), 
		// attributeBuffers
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
