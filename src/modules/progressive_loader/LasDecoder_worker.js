
import {Vector3} from "../../math/math.js";
import {Geometry} from "../../core/Geometry.js";

onmessage = async function(e){

	console.log("test");

	let {file} = e.data;

	let buffer = await file.slice(0, 375).arrayBuffer()

	let view = new DataView(buffer);
	let versionMajor = view.getUint8(24);
	let versionMinor = view.getUint8(25);

	let numPoints = view.getUint32(107, true);
	if(versionMajor >= 1 && versionMinor >= 4){
		numPoints = Number(view.getBigInt64(247, true));
	}

	let offsetToPointData = view.getUint32(96, true);
	let recordLength = view.getUint16(105, true);
	let pointFormat = view.getUint8(104);

	let scale = new Vector3(
		view.getFloat64(131, true),
		view.getFloat64(139, true),
		view.getFloat64(147, true),
	);

	let offset = new Vector3(
		view.getFloat64(155, true),
		view.getFloat64(163, true),
		view.getFloat64(171, true),
	);

	let min = new Vector3(
		view.getFloat64(187, true),
		view.getFloat64(203, true),
		view.getFloat64(219, true),
	);

	let max = new Vector3(
		view.getFloat64(179, true),
		view.getFloat64(195, true),
		view.getFloat64(211, true),
	);

	let header = {
		versionMajor, versionMinor, 
		numPoints, pointFormat, recordLength, offsetToPointData,
		min, max, scale, offset,
	};


	{
		let n = Math.min(numPoints, 1000_000);

		let buffer = await file.slice(
			offsetToPointData, 
			offsetToPointData + n * recordLength,
		).arrayBuffer()
		let view = new DataView(buffer);

		let offsetRGB = {
			"2": 20,
			"3": 28,
			"5": 28,
		}[pointFormat];

		
		let geometry = new Geometry();
		geometry.numElements = n;
		let position = new Float32Array(3 * n);
		let color = new Uint8Array(4 * n);
		for(let i = 0; i < n; i++){

			let pointOffset = i * recordLength;
			let X = view.getInt32(pointOffset + 0, true);
			let Y = view.getInt32(pointOffset + 4, true);
			let Z = view.getInt32(pointOffset + 8, true);

			let x = X * header.scale.x + header.offset.x;
			let y = Y * header.scale.y + header.offset.y;
			let z = Z * header.scale.z + header.offset.z;

			position[3 * i + 0] = x;
			position[3 * i + 1] = y;
			position[3 * i + 2] = z;

			color[4 * i + 0] = view.getUint16(pointOffset + offsetRGB + 0);
			color[4 * i + 1] = view.getUint16(pointOffset + offsetRGB + 2);
			color[4 * i + 2] = view.getUint16(pointOffset + offsetRGB + 4);
			color[4 * i + 3] = 255;
		}

		postMessage({
			numPoints: n,
			buffers: {
				position: position,
				color: color,
			},
			min, max,
		});
		
	}

	// postMessage({header});

};