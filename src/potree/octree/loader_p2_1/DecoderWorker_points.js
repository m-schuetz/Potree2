function round4(number){
	return number + (4 - (number % 4));
}

onmessage = async function (event) {

	// debugger;

	let data = event.data;
	let n    = event.data.numElements;

	let bufferSize = round4(18 * n);
	let buffer = new ArrayBuffer(bufferSize);

	let first = data.byteOffset;
	let last = first + data.byteSize - 1;

	let response = await fetch(data.url, {
		headers: {
			'content-type': 'multipart/byteranges',
			'Range': `bytes=${first}-${last}`,
		},
	});
	let source = await response.arrayBuffer();
	let sourceView = new DataView(source);
	let targetView = new DataView(buffer);
	let offset_xyz = 0;
	let offset_rgb = 12 * n;

	for(let i = 0; i < n; i++){
		let x = sourceView.getFloat32(16 * i + 0, true);
		let y = sourceView.getFloat32(16 * i + 4, true);
		let z = sourceView.getFloat32(16 * i + 8, true);
		let r = sourceView.getUint8(16 * i + 12);
		let g = sourceView.getUint8(16 * i + 13);
		let b = sourceView.getUint8(16 * i + 14);

		targetView.setFloat32(offset_xyz + 12 * i + 0, x, true);
		targetView.setFloat32(offset_xyz + 12 * i + 4, y, true);
		targetView.setFloat32(offset_xyz + 12 * i + 8, z, true);
		targetView.setUint16(offset_rgb + 6 * i + 0, r, true);
		targetView.setUint16(offset_rgb + 6 * i + 2, g, true);
		targetView.setUint16(offset_rgb + 6 * i + 4, b, true);
	}

	let message = {
		type: "points parsed", 
		buffer
	};
	let transferables = [buffer];

	postMessage(message, transferables);

};
