
import {SceneNode} from "../../scene/SceneNode.js";
import {Vector3, Box3, Matrix4} from "../../math/math.js";


class ProgressiveItem{

	constructor(){

	}

}

export class ProgressivePointCloud extends SceneNode{

	constructor({files}){

		super("progressive point cloud");

		this.files = files;

		this.items = [];

		this.renderables = {
			boxes: [],
			pointclouds: [],
		};

		this.spawnWorker();
	}

	spawnWorker(){
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

			this.renderables.boxes.push(...newDeserializedBoxes);
		};

		worker.postMessage({files: this.files});
	}

	update(){

	}

};