
import {Vector3, Box3} from "../../math/math.js";
import {Geometry} from "../../core/Geometry.js";
import {SceneNode} from "../../scene/SceneNode.js";
import {WorkerPool} from "../../misc/WorkerPool.js";

let octree = new SceneNode("octree");
octree.visibleNodes = [];
octree.updateVisibility = () => {};

let progress = {
	header: null,
	nodes: [],
};

progress.octree = octree;

let workerPath = "./src/modules/progressive_loader/LasDecoder_worker.js";
let workers = [
	WorkerPool.getWorker(workerPath, {type: "module"}),
	WorkerPool.getWorker(workerPath, {type: "module"}),
	WorkerPool.getWorker(workerPath, {type: "module"}),
	WorkerPool.getWorker(workerPath, {type: "module"}),
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
		node.boundingBox = new Box3();
		node.boundingBox.min.copy(min);
		node.boundingBox.max.copy(max);
		node.level = 0;
		node.geometry = geometry;
		
		octree.visibleNodes.push(node);

		console.log("node loaded");

		workers.push(worker);

		console.log("time: ", performance.now());
	};

	let octree_min = octree.boundingBox.min;
	worker.postMessage({file, octree_min});


}

function load(file){
	setTimeout(loadHeader, 1, file);
}


let boxes = [];

function install(element, args){

	element.addEventListener('dragover', function(e) {
		e.stopPropagation();
		e.preventDefault();
		e.dataTransfer.dropEffect = 'copy';
	});

	element.addEventListener('drop', async function(e) {
		console.log("abc");
		e.stopPropagation();
		e.preventDefault();

		let files = e.dataTransfer.files;

		let workerPath = `${import.meta.url}/../progressive_worker.js`;
		let worker = new Worker(workerPath, {type: "module"});

		worker.onmessage = (e) => {
			let newBoxes = e.data.boxes;

			let newDeserializedBoxes = newBoxes.map(newBox => {
				let boundingBox = new Box3(
					new Vector3().copy(newBox.boundingBox.min),
					new Vector3().copy(newBox.boundingBox.max),
				);
				let color = new Vector3().copy(newBox.color);

				return {boundingBox, color};
			});

			boxes.push(...newDeserializedBoxes);

			if(args.progress){
				args.progress({
					completed: boxes.length,
					total: files.length,
					newBoxes: newDeserializedBoxes,
					allBoxes: boxes,
				});
			}
		};

		worker.postMessage({
			files
		});

		boxes = [];
		let progress = {};

		if(args.init){
			args.init({boxes, progress});
		}

		

		// console.log("start: ", performance.now());

		// let promises = [];
		// for (let file of files) {
		// 	let blob = file.slice(0, 227);
		// 	let promise = blob.arrayBuffer();

		// 	promises.push(promise);
		// }


		// let buffers = await Promise.all(promises);

		// let full_aabb = new Box3();

		// let boxes = [];
		// for(let buffer of buffers){
		// 	let view = new DataView(buffer);

		// 	let min = new Vector3();
		// 	min.x = view.getFloat64(187, true);
		// 	min.y = view.getFloat64(203, true);
		// 	min.z = view.getFloat64(219, true);

		// 	let max = new Vector3();
		// 	max.x = view.getFloat64(179, true);
		// 	max.y = view.getFloat64(195, true);
		// 	max.z = view.getFloat64(211, true);

		// 	let box = new Box3(min, max);
		// 	boxes.push(box);

		// 	full_aabb.expandByPoint(min);
		// 	full_aabb.expandByPoint(max);
		// }

		// progress.boundingBox = full_aabb;

		// octree.boundingBox = full_aabb;
		// octree.position.copy(full_aabb.min);
		// octree.updateWorld();

		// // for (let file of files) {
		// // 	load(file);
		// // }

		// callback({boxes, progress});


	});
}


export {load, install};