
import {Vector3, Box3} from "../../math/math.js";


async function loadBox(file){
	
	let blob = file.slice(0, Math.min(file.size, 1_000_000));
	//let blob = file.slice(0, 227);

	let buffer = await blob.arrayBuffer();
	let view = new DataView(buffer);

	let versionMajor = view.getUint8(24);
	let versionMinor = view.getUint8(25);
	let offsetToPointData = view.getUint32(96, true);
	let formatID = view.getUint8(104);
	let recordLength = view.getUint16(105, true);
	let compressed = formatID >= 64;

	let numPoints = view.getUint32(107, true);
	if(versionMajor >= 1 && versionMinor >= 4){
		numPoints = Number(view.getBigInt64(247, true));
	}


	let min = new Vector3();
	min.x = view.getFloat64(187, true);
	min.y = view.getFloat64(203, true);
	min.z = view.getFloat64(219, true);

	let max = new Vector3();
	max.x = view.getFloat64(179, true);
	max.y = view.getFloat64(195, true);
	max.z = view.getFloat64(211, true);


	let color; 
	{
		let offsetFirstPoint = offsetToPointData;

		if(compressed){
			offsetFirstPoint += 8;
			formatID = formatID & 0b1111;

		}

		let offsetRGB = {
			0: 0,
			1: 0,
			2: 20,
			3: 28,
		}[formatID];

		let [R, G, B] = [0, 0, 0];

		let n = Math.min(numPoints, 10000);

		if(compressed){
			n = 1;
		}

		for(let i = 0; i < n; i++){
			R += view.getUint16(offsetFirstPoint + i * recordLength + offsetRGB + 0, true);
			G += view.getUint16(offsetFirstPoint + i * recordLength + offsetRGB + 2, true);
			B += view.getUint16(offsetFirstPoint + i * recordLength + offsetRGB + 4, true);
		}

		R = R / n;
		G = G / n;
		B = B / n;

		// console.log(R, G, B);

		color = new Vector3(R, G, B).multiplyScalar(1 / 256);
	}


	let box = {
		boundingBox: new Box3(min, max),
		color: color,
	}

	return box;
}


onmessage = async function(e){

	let {files} = e.data;

	for(let file of files){

		let box = await loadBox(file);

		postMessage({
			boxes: [box],
		});

	}

}

