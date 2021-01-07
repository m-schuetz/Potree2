
import {SceneNode} from "../scene/SceneNode.js";
import {PointCloudOctreeNode} from "./PointCloudOctreeNode.js";
import {BinaryHeap} from "../../libs/BinaryHeap/BinaryHeap.js";
import {toRadians} from "../math/PMath.js";
import {Matrix4} from "../math/Matrix4.js";
import {Vector3} from "../math/Vector3.js";

export class PointCloudOctree extends SceneNode{

	constructor(name){
		super(name);

		this.loader = null;
		this.root = null;
		this.spacing = 1;
		this.loaded = false;
		this.loading = false;
		this.visibleNodes = [];
	}

	load(node){

		if(!node.loaded){
			this.loader.loadNode(node);
		}

	}

	updateVisibility(camera){

		let tStart = performance.now();

		let visibleNodes = [];
		let loadQueue = [];
		let priorityQueue = new BinaryHeap(function (x) { return 1 / x.weight; });

		priorityQueue.push({ 
			node: this.root, 
			weight: Number.MAX_VALUE,
		});

		let i = 0;
		let numPoints = 0;

		let camWorld = new Matrix4();
		camWorld.elements.set(camera.world);
		let camPos = new Vector3().applyMatrix4(camWorld);

		while (priorityQueue.size() > 0) {
			let element = priorityQueue.pop();
			let {node, weight} = element;

			if(!node.loaded){
				loadQueue.push(node);

				if(loadQueue.length > 10){
					break;
				}

				continue;
			}
			if(numPoints + node.numPoints > this.pointBudget){
				break;
			}

			visibleNodes.push(node);
			numPoints += node.numPoints;


			for(let child of node.children){
				if(!child){
					continue;
				}

				let center = node.boundingBox.center();
				
				center.applyMatrix4(this.world);

				let radius = node.boundingBox.min.distanceTo(node.boundingBox.max) / 2;

				let dx = camPos.x - center.x;
				let dy = camPos.y - center.y;
				let dz = camPos.z - center.z;

				let dd = dx * dx + dy * dy + dz * dz;
				let distance = Math.sqrt(dd);


				let fov = toRadians(camera.fov);
				let slope = Math.tan(fov / 2);
				let projFactor = 1 / (slope * distance);
				// let projFactor = (0.5 * domHeight) / (slope * distance);

				let weight = radius * projFactor;

				if(distance - radius < 0){
					weight = Number.MAX_VALUE;
				}

				priorityQueue.push({
					node: child, 
					weight: weight
				});
			}

		}

		for(let node of loadQueue){
			this.load(node);
		}


		this.visibleNodes = visibleNodes;

		let duration = 1000 * (performance.now() - tStart);

		return duration;
	}

	updateVisibility1(camera){

		let visibleNodes = [];

		// traverse breadth first
		let loadQueue = [];
		let queue = [this.root];
		while(queue.length > 0){
			let node = queue.shift();

			if(!node.loaded){
				loadQueue.push(node);

				if(loadQueue.length > 10){
					break;
				}

				continue;
			}

			visibleNodes.push(node);

			if(node.level < 4){
				for(let child of node.children){
					if(child){
						queue.push(child);
					}
				}
			}

		}

		for(let node of loadQueue){
			this.load(node);
		}


		this.visibleNodes = visibleNodes;

		// if(loadQueue.length >= 4){
			
		// 	loadQueue.sort((a, b) => {

		// 		if(a.byteOffset == null || b.byteOffset == null){
		// 			return -1;
		// 		}

		// 		return Number(a.byteOffset - b.byteOffset);
		// 	});

		// 	let first = {
		// 		byteOffset: loadQueue[0].byteOffset,
		// 		byteSize: loadQueue[0].byteSize,
		// 		nodes: [loadQueue[0]],
		// 	};
		// 	let batches = [first];
		// 	for(let i = 1; i < loadQueue.length; i++){
				
		// 		let a = batches[batches.length - 1];
		// 		let b = first = {
		// 			byteOffset: loadQueue[i].byteOffset,
		// 			byteSize: loadQueue[i].byteSize,
		// 			nodes: [loadQueue[i]],
		// 		};

		// 		if(a.byteOffset == null || b.byteOffset == null){
		// 			continue;
		// 		}
				
		// 		// if(a.byteOffset + a.byteSize === b.byteOffset){
		// 		if(Math.abs(Number((a.byteOffset + a.byteSize) - b.byteOffset)) < 100_000 ){
		// 			// merge
		// 			a.byteSize += b.byteSize;
		// 			a.nodes.push(b);
		// 		}else{
		// 			batches.push(b);
		// 		}
				
				
		// 	}
		// 	console.log(`${loadQueue.length} => ${batches.length}`);
		// }

		
	}

}