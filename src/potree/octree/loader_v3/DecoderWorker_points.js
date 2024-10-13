
// import {BrotliDecode} from "../../../../libs/brotli/decode.js";

function round4(number){
	return number + (4 - (number % 4));
}

async function loadNode(event){
	let data = event.data;
	let n    = event.data.numElements;
	let {scale, offset} = data;

	let bufferSize = round4(18 * n);
	let buffer = new ArrayBuffer(bufferSize);

	let first = event.data.metadata.pointBuffer.offset + data.byteOffset;
	let last = first + data.byteSize - 1;

	let response = await fetch(data.url, {
		headers: {
			'content-type': 'multipart/byteranges',
			'Range': `bytes=${first}-${last}`,
		},
	});

	{
		let nmin = data.nodeMin;
		let nmax = data.nodeMax;
		let nsize = nmax[0] - nmin[0];
		let cellsize = nsize / 128;
		let precision = 0.001;
		let states = cellsize / precision;
		let bits = (Math.log2(states)).toFixed(1);

		// console.log(`leaf, size: ${nsize}, cellsize: ${cellsize.toFixed(3)}, bits: ${bits}`);

	}

	let s_stride = data.byteSize / n;

	let source = await response.arrayBuffer();
	let sourceView = new DataView(source);
	let targetView = new DataView(buffer);
	let t_offset_xyz = 0;
	let t_offset_rgb = 12 * n;
	let s_offset_rgb = 0;
	for(let attribute of data.pointAttributes.attributes){
		
		if(["rgb", "rgba"].includes(attribute.name)){
			s_offset_rgb = attribute.byteOffset;
			break;
		}

	}

	for(let i = 0; i < n; i++){
		let X = sourceView.getInt32(s_stride * i + 0, true);
		let Y = sourceView.getInt32(s_stride * i + 4, true);
		let Z = sourceView.getInt32(s_stride * i + 8, true);

		let x = X * scale[0] + offset[0] - data.min[0];
		let y = Y * scale[1] + offset[1] - data.min[1];
		let z = Z * scale[2] + offset[2] - data.min[2];

		// let r = sourceView.getUint8(s_stride * i + 12);
		// let g = sourceView.getUint8(s_stride * i + 13);
		// let b = sourceView.getUint8(s_stride * i + 14);
		let R = sourceView.getUint16(s_stride * i + s_offset_rgb + 0, true);
		let G = sourceView.getUint16(s_stride * i + s_offset_rgb + 2, true);
		let B = sourceView.getUint16(s_stride * i + s_offset_rgb + 4, true);

		let r = R < 256 ? R : R / 256;
		let g = G < 256 ? G : G / 256;
		let b = B < 256 ? B : B / 256;
		// debugger;

		targetView.setFloat32(t_offset_xyz + 12 * i + 0, x, true);
		targetView.setFloat32(t_offset_xyz + 12 * i + 4, y, true);
		targetView.setFloat32(t_offset_xyz + 12 * i + 8, z, true);
		targetView.setUint16(t_offset_rgb + 6 * i + 0, r, true);
		targetView.setUint16(t_offset_rgb + 6 * i + 2, g, true);
		targetView.setUint16(t_offset_rgb + 6 * i + 4, b, true);
	}

	let message = {
		type: "points parsed", 
		buffer
	};
	let transferables = [buffer];

	postMessage(message, transferables);
}

onmessage = async function (event) {

	let promise = loadNode(event);

	// Chrome frequently fails with range requests.
	// Notify main thread that loading failed, so that it can try again.
	promise.catch(e => {
		console.log(e);
		postMessage("failed");
	});

};
