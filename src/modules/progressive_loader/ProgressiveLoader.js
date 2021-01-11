
import {Vector3, Box3} from "../../math/math.js";
import {Geometry} from "../../core/Geometry.js";
import {SceneNode} from "../../scene/SceneNode.js";
import {WorkerPool} from "../../misc/WorkerPool.js";

let octree = new SceneNode("octree");
octree.visibleNodes = [];

let progress = {
	header: null,
	nodes: [],
};

progress.octree = octree;

async function loadHeader(file){

	let workerPath = "./src/modules/progressive_loader/LasDecoder_worker.js";
	let worker = WorkerPool.getWorker(workerPath, {type: "module"});

	worker.onmessage = (e) => {
		let {buffers, numPoints, min, max} = e.data;

		let geometry = new Geometry();
		geometry.numElements = numPoints;

		geometry.buffers.push({
			name: "position",
			buffer: buffers.position
		});

		geometry.buffers.push({
			name: "color",
			buffer: buffers.color
		});

		let node = new SceneNode("node 0");
		
		node.name = "0";
		node.loaded = true;
		node.boundingBox = new Box3(min, max);
		node.level = 0;
		node.geometry = geometry;
		
		octree.visibleNodes.push(node);

		console.log("node loaded");
	};
	worker.postMessage({file});

	// let buffer = await file.slice(0, 375).arrayBuffer()

	// let view = new DataView(buffer);
	// let versionMajor = view.getUint8(24);
	// let versionMinor = view.getUint8(25);

	// let numPoints = view.getUint32(107, true);
	// if(versionMajor >= 1 && versionMinor >= 4){
	// 	numPoints = Number(view.getBigInt64(247, true));
	// }

	// let offsetToPointData = view.getUint32(96, true);
	// let recordLength = view.getUint16(105, true);
	// let pointFormat = view.getUint8(104);

	// let scale = new Vector3(
	// 	view.getFloat64(131, true),
	// 	view.getFloat64(139, true),
	// 	view.getFloat64(147, true),
	// );

	// let offset = new Vector3(
	// 	view.getFloat64(155, true),
	// 	view.getFloat64(163, true),
	// 	view.getFloat64(171, true),
	// );

	// let min = new Vector3(
	// 	view.getFloat64(187, true),
	// 	view.getFloat64(203, true),
	// 	view.getFloat64(219, true),
	// );

	// let max = new Vector3(
	// 	view.getFloat64(179, true),
	// 	view.getFloat64(195, true),
	// 	view.getFloat64(211, true),
	// );

	// let header = {
	// 	versionMajor, versionMinor, 
	// 	numPoints, pointFormat, recordLength, offsetToPointData,
	// 	min, max, scale, offset,
	// };

	// progress.header = header;

	// {
	// 	let buffer = await file.slice(offsetToPointData, file.size).arrayBuffer()
	// 	let view = new DataView(buffer);

	// 	let offsetRGB = {
	// 		"2": 20,
	// 		"3": 28,
	// 		"5": 28,
	// 	}[pointFormat];

	// 	let n = Math.min(numPoints, 100_000);
	// 	let geometry = new Geometry();
	// 	geometry.numElements = n;
	// 	let position = new Float32Array(3 * n);
	// 	let color = new Uint8Array(4 * n);
	// 	for(let i = 0; i < n; i++){

	// 		let pointOffset = i * recordLength;
	// 		let X = view.getInt32(pointOffset + 0, true);
	// 		let Y = view.getInt32(pointOffset + 4, true);
	// 		let Z = view.getInt32(pointOffset + 8, true);

	// 		let x = X * header.scale.x + header.offset.x;
	// 		let y = Y * header.scale.y + header.offset.y;
	// 		let z = Z * header.scale.z + header.offset.z;

	// 		position[3 * i + 0] = x;
	// 		position[3 * i + 1] = y;
	// 		position[3 * i + 2] = z;

	// 		color[4 * i + 0] = view.getUint16(pointOffset + offsetRGB + 0);
	// 		color[4 * i + 1] = view.getUint16(pointOffset + offsetRGB + 2);
	// 		color[4 * i + 2] = view.getUint16(pointOffset + offsetRGB + 4);
	// 		color[4 * i + 3] = 255;
	// 	}
	// 	geometry.buffers.push({
	// 		name: "position",
	// 		buffer: position
	// 	});

	// 	geometry.buffers.push({
	// 		name: "color",
	// 		buffer: color
	// 	});

	// 	let node = new SceneNode("node 0");
		
	// 	node.name = "0";
	// 	node.loaded = true;
	// 	node.boundingBox = new Box3(min, max);
	// 	node.level = 0;
	// 	node.geometry = geometry;
		
	// 	octree.visibleNodes.push(node);
	// }

}

function load(file){
	setTimeout(loadHeader, 1, file);
}


function install(element, callback){

	element.addEventListener('dragover', function(e) {
		e.stopPropagation();
		e.preventDefault();
		e.dataTransfer.dropEffect = 'copy';
	});

	element.addEventListener('drop', async function(e) {
		e.stopPropagation();
		e.preventDefault();

		let files = e.dataTransfer.files;

		let promises = [];
		for (let file of files) {
			let blob = file.slice(0, 227);
			let promise = blob.arrayBuffer();

			promises.push(promise);

			load(file);
		}


		let buffers = await Promise.all(promises);

		let boxes = [];
		for(let buffer of buffers){
			let view = new DataView(buffer);

			let min = new Vector3();
			min.x = view.getFloat64(187, true);
			min.y = view.getFloat64(203, true);
			min.z = view.getFloat64(219, true);

			let max = new Vector3();
			max.x = view.getFloat64(179, true);
			max.y = view.getFloat64(195, true);
			max.z = view.getFloat64(211, true);

			let box = new Box3(min, max);
			boxes.push(box);
		}

		// load(files.item(0));
		// load(files.item(1));
		// load(files.item(2));

		callback({boxes, progress});


	});
}


export {load, install};