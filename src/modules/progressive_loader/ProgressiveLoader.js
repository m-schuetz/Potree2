
import {Vector3, Box3} from "../../math/math.js";
import {Geometry} from "../../core/Geometry.js";
import {SceneNode} from "../../scene/SceneNode.js";
import {WorkerPool} from "../../misc/WorkerPool.js";
import {ProgressivePointCloud} from "./ProgressivePointCloud.js";

// let workerPath = "./src/modules/progressive_loader/LasDecoder_worker.js";
// let workers = [
// 	WorkerPool.getWorker(workerPath, {type: "module"}),
// 	WorkerPool.getWorker(workerPath, {type: "module"}),
// 	WorkerPool.getWorker(workerPath, {type: "module"}),
// 	WorkerPool.getWorker(workerPath, {type: "module"}),
// 	// WorkerPool.getWorker(workerPath, {type: "module"}),
// 	// WorkerPool.getWorker(workerPath, {type: "module"}),
// ];


// async function loadHeader(file){

// 	if(workers.length == 0){
// 		setTimeout(loadHeader, 1, file);

// 		return;
// 	}

// 	let worker = workers.pop();

// 	worker.onmessage = (e) => {
// 		let {buffers, numPoints, min, max} = e.data;

// 		let geometry = new Geometry();
// 		geometry.numElements = numPoints;

// 		geometry.buffers.push({
// 			name: "position",
// 			buffer: buffers.position
// 		});

// 		geometry.buffers.push({
// 			name: "color",
// 			buffer: buffers.color
// 		});

// 		let node = new SceneNode("node 0");
		
// 		node.name = "0";
// 		node.loaded = true;
// 		node.boundingBox = new Box3();
// 		node.boundingBox.min.copy(min);
// 		node.boundingBox.max.copy(max);
// 		node.level = 0;
// 		node.geometry = geometry;
		
// 		octree.visibleNodes.push(node);

// 		console.log("node loaded");

// 		workers.push(worker);

// 		console.log("time: ", performance.now());
// 	};

// 	let octree_min = octree.boundingBox.min;
// 	worker.postMessage({file, octree_min});


// }

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

		let progressivePointCloud = new ProgressivePointCloud({files});

		if(args.progressivePointcloudAdded){
			args.progressivePointcloudAdded({node: progressivePointCloud});
		}

	});
}


export {install};