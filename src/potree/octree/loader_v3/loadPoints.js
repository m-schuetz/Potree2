import {BrotliDecode} from "../../../../libs/brotli/decode.js";

// round up to nearest <n>
function ceil_n(number, n){
	return number + (n - (number % n));
}

export function loadPointsBrotli(octree, node, dataview){

	let n = node.numPoints;
	let {scale, offset} = octree;

	let attributes = octree.pointAttributes;
	let bytesPerPoint = octree.pointAttributes.byteSize;

	let bufferSize = ceil_n(bytesPerPoint * n, 4);
	let buffer = new ArrayBuffer(bufferSize);

	// let s_stride = node.byteSize / n;
	let s_stride = bytesPerPoint;
	let targetView = new DataView(buffer);
	let t_offset_xyz = 0;

	let compressedBuffer = new Int8Array(dataview.buffer, dataview.byteOffset, dataview.byteLength);

	// let t_start = Date.now();
	let decoded = BrotliDecode(compressedBuffer);
	// let t_end = Date.now();
	// let milliseconds = t_end - t_start;
	// let pointsPerSec = Math.floor((n / milliseconds) * 1000).toLocaleString();
	// console.log(`#points: ${n} in ${milliseconds.toFixed(1)} ms. points/sec: ${pointsPerSec}`);
	let view = new DataView(decoded.buffer);

	new Uint8Array(buffer).set(decoded);

	// decode int32 encoded positions to float positions
	for(let i = 0; i < n; i++){
		let X = view.getInt32(12 * i + 0, true);
		let Y = view.getInt32(12 * i + 4, true);
		let Z = view.getInt32(12 * i + 8, true);

		let x = X * scale[0] + offset[0] - octree.min[0];
		let y = Y * scale[1] + offset[1] - octree.min[1];
		let z = Z * scale[2] + offset[2] - octree.min[2];

		targetView.setFloat32(t_offset_xyz + 12 * i + 0, x, true);
		targetView.setFloat32(t_offset_xyz + 12 * i + 4, y, true);
		targetView.setFloat32(t_offset_xyz + 12 * i + 8, z, true);
	}


	// NOTE: old code that was used to transform from array-of-struct order to struct-of-array order
	// Now it's already stored in the expected order.
	// 
	// copy unfiltered attributes
	// let source_u8 = new Uint8Array(dataview.buffer);
	// let target_u8 = new Uint8Array(buffer);
	// for(let attribute of attributes.attributes){

	// 	if(attribute.name === "position") continue;
	// 	// if(attribute.name === "rgba") continue;

	// 	for(let pointIndex = 0; pointIndex < n; pointIndex++)
	// 	{

	// 		for(let j = 0; j < attribute.byteSize; j++){
	// 			// let value = source_u8[s_stride * pointIndex + attribute.byteOffset + j];
	// 			let value = view.getUint8(s_stride * pointIndex + attribute.byteOffset + j);
	// 			target_u8[n * attribute.byteOffset + pointIndex * attribute.byteSize + j] = value;
	// 		}
	// 	}
	// }

	return {buffer};

}

export function loadPoints(octree, node, dataview){

	let n = node.numPoints;
	let {scale, offset} = octree;

	let attributes = octree.pointAttributes;
	let bytesPerPoint = octree.pointAttributes.byteSize;

	let bufferSize = ceil_n(bytesPerPoint * n, 4);
	let buffer = new ArrayBuffer(bufferSize);

	let s_stride = node.byteSize / n;
	let targetView = new DataView(buffer);
	let t_offset_xyz = 0;

	// decode int32 encoded positions to float positions
	for(let i = 0; i < n; i++){
		let X = dataview.getInt32(s_stride * i + 0, true);
		let Y = dataview.getInt32(s_stride * i + 4, true);
		let Z = dataview.getInt32(s_stride * i + 8, true);

		let x = X * scale[0] + offset[0] - octree.min[0];
		let y = Y * scale[1] + offset[1] - octree.min[1];
		let z = Z * scale[2] + offset[2] - octree.min[2];

		targetView.setFloat32(t_offset_xyz + 12 * i + 0, x, true);
		targetView.setFloat32(t_offset_xyz + 12 * i + 4, y, true);
		targetView.setFloat32(t_offset_xyz + 12 * i + 8, z, true);
	}


	// copy unfiltered attributes
	let source_u8 = new Uint8Array(dataview.buffer);
	let target_u8 = new Uint8Array(buffer);
	for(let attribute of attributes.attributes){

		if(attribute.name === "position") continue;
		// if(attribute.name === "rgba") continue;

		for(let pointIndex = 0; pointIndex < n; pointIndex++)
		{

			for(let j = 0; j < attribute.byteSize; j++){
				// let value = source_u8[s_stride * pointIndex + attribute.byteOffset + j];
				let value = dataview.getUint8(s_stride * pointIndex + attribute.byteOffset + j);
				target_u8[n * attribute.byteOffset + pointIndex * attribute.byteSize + j] = value;
			}
		}
	}

	return {buffer};

}