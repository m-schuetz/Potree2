
import {PointAttribute, PointAttributes, PointAttributeTypes} from "./PointAttributes.js";
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

onmessage = function (event) {

	//let {pointAttributes, scale, name, min, max, size, offset, numPoints} = event.data;
	let {pointAttributes, numPoints, scale, offset, min} = event.data;

	let tStart = performance.now();

	let buffer = BrotliDecode(new Int8Array(event.data.buffer));
	let view = new DataView(buffer.buffer);
	
	let attributeBuffers = {};

	let bytesPerPoint = 0;
	for (let pointAttribute of pointAttributes.attributes) {
		bytesPerPoint += pointAttribute.byteSize;
	}

	let byteOffset = 0;
	for (let pointAttribute of pointAttributes.attributes) {
		

		if(["POSITION_CARTESIAN", "position"].includes(pointAttribute.name)){

			// let tStart = performance.now();

			let buff = new ArrayBuffer(numPoints * 4 * 3);
			let positions = new Float32Array(buff);
		
			for (let j = 0; j < numPoints; j++) {


				let mc_0 = view.getUint32(byteOffset +  4, true);
				let mc_1 = view.getUint32(byteOffset +  0, true);
				let mc_2 = view.getUint32(byteOffset + 12, true);
				let mc_3 = view.getUint32(byteOffset +  8, true);

				byteOffset += 16;

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

				positions[3 * j + 0] = x;
				positions[3 * j + 1] = y;
				positions[3 * j + 2] = z;
			}

			// let duration = performance.now() - tStart;
			// console.log(`xyz: ${duration.toFixed(1)}ms`);

			attributeBuffers[pointAttribute.name] = { buffer: positions, attribute: pointAttribute };
			// attributeBuffers[pointAttribute.name] = { buffer: buff, attribute: pointAttribute };
		}else if(["RGBA", "rgba"].includes(pointAttribute.name)){

			let buff = new ArrayBuffer(numPoints * 4);
			let colors = new Uint8Array(buff);

			// for (let j = 0; j < numPoints; j++) {
			// 	let r = view.getUint16(byteOffset + 0, true);
			// 	let g = view.getUint16(byteOffset + 2, true);
			// 	let b = view.getUint16(byteOffset + 4, true);
			// 	byteOffset += 6;

			// 	colors[4 * j + 0] = r > 255 ? r / 256 : r;
			// 	colors[4 * j + 1] = g > 255 ? g / 256 : g;
			// 	colors[4 * j + 2] = b > 255 ? b / 256 : b;
			// }

			// let tStart = performance.now();

			for (let j = 0; j < numPoints; j++) {

				let mc_0 = view.getUint32(byteOffset +  4, true);
				let mc_1 = view.getUint32(byteOffset +  0, true);
				byteOffset += 8;

				let r = dealign24b((mc_1 & 0x00FFFFFF) >>> 0) 
						| (dealign24b(((mc_1 >>> 24) | (mc_0 << 8)) >>> 0) << 8);

				let g = dealign24b((mc_1 & 0x00FFFFFF) >>> 1) 
						| (dealign24b(((mc_1 >>> 24) | (mc_0 << 8)) >>> 1) << 8);

				let b = dealign24b((mc_1 & 0x00FFFFFF) >>> 2) 
						| (dealign24b(((mc_1 >>> 24) | (mc_0 << 8)) >>> 2) << 8);

				// let bits = mask_b0[mc_1 >>> 24];
				
				// if(((r >> 8) & 0b11) !== bits){
				// 	debugger;	
				// }

				// let r = dealign24b(mc0 >> 0) | (dealign24b(mc1 >> 0) << 8);
				// let g = dealign24b(mc0 >> 1) | (dealign24b(mc1 >> 1) << 8);
				// let b = dealign24b(mc0 >> 2) | (dealign24b(mc1 >> 2) << 8);


				colors[4 * j + 0] = r > 255 ? r / 256 : r;
				colors[4 * j + 1] = g > 255 ? g / 256 : g;
				colors[4 * j + 2] = b > 255 ? b / 256 : b;
				colors[4 * j + 3] = 255;
			}
			// let duration = performance.now() - tStart;
			// console.log(`rgb: ${duration.toFixed(1)}ms`);

			attributeBuffers[pointAttribute.name] = { buffer: colors, attribute: pointAttribute };
			// attributeBuffers[pointAttribute.name] = { buffer: buff, attribute: pointAttribute };
		}else{
			let buff = new ArrayBuffer(numPoints * 4);
			let f32 = new Float32Array(buff);

			let TypedArray = typedArrayMapping[pointAttribute.type.name];
			preciseBuffer = new TypedArray(numPoints);

			let [offset, scale] = [0, 1];

			const getterMap = {
				"int8":   view.getInt8,
				"int16":  view.getInt16,
				"int32":  view.getInt32,
				// "int64":  view.getInt64,
				"uint8":  view.getUint8,
				"uint16": view.getUint16,
				"uint32": view.getUint32,
				// "uint64": view.getUint64,
				"float":  view.getFloat32,
				"double": view.getFloat64,
			};
			const getter = getterMap[pointAttribute.type.name].bind(view);

			// compute offset and scale to pack larger types into 32 bit floats
			if(pointAttribute.type.size > 4){
				let [amin, amax] = pointAttribute.range;
				offset = amin;
				scale = 1 / (amax - amin);
			}

			for(let j = 0; j < numPoints; j++){
				// let pointOffset = j * bytesPerPoint;
				let value = getter(byteOffset, true);
				byteOffset += pointAttribute.byteSize;

				f32[j] = (value - offset) * scale;
				preciseBuffer[j] = value;
			}

			attributeBuffers[pointAttribute.name] = { 
				buffer: buff,
				preciseBuffer: preciseBuffer,
				attribute: pointAttribute,
				offset: offset,
				scale: scale,
			};
		}

		// attributeOffset += pointAttribute.byteSize;
	}

	let duration = performance.now() - tStart;
	let pointsPerMs = numPoints / duration;
	console.log(`duration: ${duration.toFixed(1)}ms, #points: ${numPoints}, points/ms: ${pointsPerMs.toFixed(1)}`);

	let message = {
		// buffer: buffer,
		attributeBuffers: attributeBuffers,
	};

	let transferables = [];
	for (let property in message.attributeBuffers) {

		let buffer = message.attributeBuffers[property].buffer;

		if(buffer instanceof ArrayBuffer){
			transferables.push(buffer);
		}else{
			transferables.push(buffer.buffer);
		}
	}

	postMessage(message, transferables);
};
