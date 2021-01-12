
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

let workerPath = "./src/modules/progressive_loader/LasDecoder_worker.js";
let workers = [
	WorkerPool.getWorker(workerPath, {type: "module"}),
	WorkerPool.getWorker(workerPath, {type: "module"}),
	// WorkerPool.getWorker(workerPath, {type: "module"}),
	// WorkerPool.getWorker(workerPath, {type: "module"}),
	// WorkerPool.getWorker(workerPath, {type: "module"}),
	// WorkerPool.getWorker(workerPath, {type: "module"}),
];


async function loadHeader(file){

	if(workers.length == 0){
		setTimeout(loadHeader, 1, file);

		return;
	}

	let worker = workers.pop();

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

		workers.push(worker);

		console.log("time: ", performance.now());
	};

	worker.postMessage({file});


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

		console.log("start: ", performance.now());

		let promises = [];
		for (let file of files) {
			let blob = file.slice(0, 227);
			let promise = blob.arrayBuffer();

			promises.push(promise);

			load(file);
		}


		let buffers = await Promise.all(promises);

		let full_aabb = new Box3();

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

			full_aabb.expandByPoint(min);
			full_aabb.expandByPoint(max);
		}

		progress.boundingBox = full_aabb;

		// load(files.item(0));
		// load(files.item(1));
		// load(files.item(2));

		callback({boxes, progress});


	});
}


export {load, install};