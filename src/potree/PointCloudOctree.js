
import {Vector3, Matrix4, Frustum, math, EventDispatcher} from "potree";
import {SceneNode, PointCloudOctreeNode} from "potree";
import {renderPointsOctree} from "potree";
import {BinaryHeap} from "BinaryHeap";
import {PointCloudMaterial} from "./PointCloudMaterial.js";

export class PointCloudOctree extends SceneNode{

	constructor(name){
		super(name);

		this.loader = null;
		this.root = null;
		this.spacing = 1;
		this.loaded = false;
		this.loading = false;
		this.visibleNodes = [];
		this.material = new PointCloudMaterial();

		let dispatcher = new EventDispatcher();
		this.events = {
			dispatcher,
			onRootNodeLoaded: (callback, args) => dispatcher.add("root_node_loaded", callback, args),
			onMaterialChanged: (callback, args) => dispatcher.add("material_changed", callback, args),
		};

		this.material.events.onChange((event) => this.events.dispatcher.dispatch("material_changed", {material: event.material}));
	}

	load(node){

		if(!node.loaded){
			this.loader.loadNode(node);
		}

	}

	updateVisibility(camera){

		let tStart = performance.now();

		if(!this.material.initialized && this.root?.geometry?.statsList != null){
			this.material.init(this);
		}

		let visibleNodes = [];
		let loadQueue = [];
		let priorityQueue = new BinaryHeap(function (x) { return 1 / x.weight; });

		let camPos = camera.getWorldPosition();
		let world = this.world;
		let view = camera.view;
		let proj = camera.proj;
		let fm = new Matrix4().multiply(proj).multiply(view); 
		let frustum = new Frustum();
		frustum.setFromMatrix(fm);

		priorityQueue.push({ 
			node: this.root, 
			weight: Number.MAX_VALUE,
		});

		let i = 0;
		let numPoints = 0;

		while (priorityQueue.size() > 0) {
			let element = priorityQueue.pop();
			let {node, weight} = element;

			if(!node.loaded){

				if(loadQueue.length < 10){
					loadQueue.push(node);
				}

				continue;
			}

			if(numPoints + node.numPoints > this.pointBudget){
				break;
			}

			let box = node.boundingBox.clone();
			box.applyMatrix4(this.world);
			let insideFrustum = frustum.intersectsBox(box);

			let visible = insideFrustum;
			visible = visible || node.level <= 2;
			// visible = visible && visibleNodes.length <= 1;

			if(!visible){
				continue;
			}

			visibleNodes.push(node);
			numPoints += node.numPoints;


			for(let child of node.children){
				if(!child){
					continue;
				}

				let center = box.center();
				
				let radius = box.min.distanceTo(box.max) / 2;

				let dx = camPos.x - center.x;
				let dy = camPos.y - center.y;
				let dz = camPos.z - center.z;

				let dd = dx * dx + dy * dy + dz * dz;
				let distance = Math.sqrt(dd);


				let fov = math.toRadians(camera.fov);
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

	render(drawstate){
		renderPointsOctree([this], drawstate);
	}

}