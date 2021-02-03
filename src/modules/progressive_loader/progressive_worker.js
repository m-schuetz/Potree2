
import {Vector3, Box3} from "../../math/math.js";


async function loadBox(file){
	let blob = file.slice(0, 1_000_000);
	//let blob = file.slice(0, 227);

	let buffer = await blob.arrayBuffer();
	let view = new DataView(buffer);

	let offsetToPointData = view.getUint32(96, true);
	let formatID = view.getUint8(104);
	let compressed = formatID >= 64;


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

		let R = view.getUint16(offsetFirstPoint + offsetRGB + 0, true);
		let G = view.getUint16(offsetFirstPoint + offsetRGB + 2, true);
		let B = view.getUint16(offsetFirstPoint + offsetRGB + 4, true);

		console.log(R, G, B);

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

