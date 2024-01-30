
// round up to nearest <n>
function ceil_n(number, n){
	return number + (n - (number % n));
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
	// let t_offset_rgb = 12 * n;
	// let s_offset_rgb = 0;

	// for(let attribute of octree.pointAttributes.attributes){
	// 	if(["rgb", "rgba"].includes(attribute.name)){
	// 		s_offset_rgb = attribute.byteOffset;
	// 		break;
	// 	}
	// }

	// decode int32 encoded positions to float positions
	for(let i = 0; i < n; i++){
		let X = dataview.getInt32(s_stride * i + 0, true);
		let Y = dataview.getInt32(s_stride * i + 4, true);
		let Z = dataview.getInt32(s_stride * i + 8, true);

		let x = X * scale[0] + offset[0] - octree.min[0];
		let y = Y * scale[1] + offset[1] - octree.min[1];
		let z = Z * scale[2] + offset[2] - octree.min[2];

		// let R = dataview.getUint16(s_stride * i + s_offset_rgb + 0, true);
		// let G = dataview.getUint16(s_stride * i + s_offset_rgb + 2, true);
		// let B = dataview.getUint16(s_stride * i + s_offset_rgb + 4, true);

		// let r = R < 256 ? R : R / 256;
		// let g = G < 256 ? G : G / 256;
		// let b = B < 256 ? B : B / 256;

		targetView.setFloat32(t_offset_xyz + 12 * i + 0, x, true);
		targetView.setFloat32(t_offset_xyz + 12 * i + 4, y, true);
		targetView.setFloat32(t_offset_xyz + 12 * i + 8, z, true);
		// targetView.setUint16(t_offset_rgb + 6 * i + 0, r, true);
		// targetView.setUint16(t_offset_rgb + 6 * i + 2, g, true);
		// targetView.setUint16(t_offset_rgb + 6 * i + 4, b, true);

		// debugger;
		// {// intensity
		// 	let s_offset_intensity = 18;
		// 	let t_offset_unfiltered = 18 * n;
		// 	let intensity = dataview.getUint16(s_stride * i + s_offset_intensity, true);
		// 	targetView.setUint16(t_offset_unfiltered + 2 * i, intensity, true);
		// }

		// {// classification
		// 	let s_offset = 20;
		// 	let t_offset_unfiltered = 20 * n;
		// 	let classification = dataview.getUint8(s_stride * i + s_offset);
		// 	targetView.setUint8(t_offset_unfiltered + 1 * i, classification);
		// }
	}

	// copy unfiltered attributes
	let source_u8 = new Uint8Array(dataview.buffer);
	let target_u8 = new Uint8Array(buffer);
	for(let attribute of attributes.attributes){

		debugger;

		if(attribute.name === "position") continue;
		// if(attribute.name === "rgba") continue;

		for(let pointIndex = 0; pointIndex < n; pointIndex++)
		for(let j = 0; j < attribute.byteSize; j++)
		{
			let value = source_u8[s_stride * pointIndex + attribute.byteOffset + j];
			target_u8[n * attribute.byteOffset + pointIndex * attribute.byteSize + j] = value;
		}

	}

	return {buffer};

}