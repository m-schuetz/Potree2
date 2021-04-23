
import {SceneNode} from "../../scene/SceneNode.js";
import {Vector3, Box3, Matrix4, Ray} from "../../math/math.js";


class ProgressiveItem{

	constructor({file, boundingBox, box, numPoints}){
		this.file = file;
		this.numPoints = numPoints;
		this.boundingBox = boundingBox;
		this.stage_0 = box;
	}

}

export class ProgressivePointCloud extends SceneNode{

	constructor({files}){

		super("progressive point cloud");

		this.files = files;

		this.items = [];

		this.renderables = {
			boxes: [],
			boundingBoxes: [],
			pointclouds: [],
		};

		this.spawnWorker();
	}

	spawnWorker(){
		let workerPath = `${import.meta.url}/../progressive_worker.js`;
		let worker = new Worker(workerPath, {type: "module"});

		worker.onmessage = (e) => {
			let stage0 = e.data;

			let boundingBox = new Box3(
				new Vector3().copy(stage0.boundingBox.min),
				new Vector3().copy(stage0.boundingBox.max),
			);
			let color = new Vector3().copy(stage0.color);
			let box = {boundingBox, color};

			this.renderables.boxes.push(box);

			let item = new ProgressiveItem({
				file: stage0.file, 
				numPoints: stage0.numPoints,
				boundingBox, 
				box
			});

			this.items.push(item);
		};

		worker.postMessage({files: this.files});
	}

	update(renderer, camera){

		let boundingBoxes = [];

		let ray = new Ray(
			camera.getWorldPosition(),
			camera.getWorldDirection(),
		);
		let view = camera.view;
		let aspect = camera.aspect;

		let numPoints_priority0 = 0;
		let numPoints_priority1 = 0;
		let numPoints_priority2 = 0;

		let stage0 = [];
		let stage1 = [];
		let stage2 = [];

		for(let item of this.items){

			let box = item.boundingBox;
			let center = box.center();
			let radius = box.min.distanceTo(box.max) / 2;

			let center_view = center.clone().applyMatrix4(view);
			let depth = -center_view.z;
			let center_distance = Math.sqrt(center_view.x ** 2 + (center_view.y * aspect) ** 2);
			let weight = center_distance / depth;

			if(depth + radius < 0){

				weight = 100;
			}

			if(weight < 0.2){
				boundingBoxes.push({
					boundingBox: item.boundingBox,
					color: new Vector3(0, 255, 0),
				});

				stage0.push(item);
				numPoints_priority0 += item.numPoints;
			}else if(weight < 0.5){
				boundingBoxes.push({
					boundingBox: item.boundingBox,
					color: new Vector3(255, 255, 0),
				});

				stage1.push(item);
				numPoints_priority1 += item.numPoints;
			}else{
				boundingBoxes.push({
					boundingBox: item.boundingBox,
					color: new Vector3(255, 0, 0),
				});

				stage2.push(item);
				numPoints_priority2 += item.numPoints;
			}
		}

		// this.renderables.boundingBoxes = boundingBoxes;
		// this.renderables.boundingBoxes = [];

		{ // dbg
			let msg = `
			priority 0: ${numPoints_priority0.toLocaleString()}
			priority 1: ${numPoints_priority1.toLocaleString()}
			priority 2: ${numPoints_priority2.toLocaleString()}
			`;

			// document.getElementById("big_message").innerText = msg;
		}

	}

};