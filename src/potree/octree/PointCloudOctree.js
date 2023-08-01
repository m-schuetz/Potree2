
import {Vector3, Matrix4, Box3, Frustum, math, EventDispatcher} from "potree";
import {SceneNode, PointCloudOctreeNode, PointCloudMaterial} from "potree";
import {renderPointsOctree} from "potree";
import {BinaryHeap} from "BinaryHeap";

const _box = new Box3();
const _fm = new Matrix4();

export class PointCloudOctree extends SceneNode{

	constructor(name){
		super(name);

		this.loader         = null;
		this.root           = null;
		this.spacing        = 1;
		this.loaded         = false;
		this.loading        = false;
		this.visibleNodes   = [];
		this.material       = new PointCloudMaterial();

		this.gpuBuffer      = null;

		this.dispatcher = new EventDispatcher();
		this.events = {
			dispatcher: this.dispatcher,
			onRootNodeLoaded: (callback, args) => this.dispatcher.add("root_node_loaded", callback, args),
			onMaterialChanged: (callback, args) => this.dispatcher.add("material_changed", callback, args),
		};

		this.dispatcher.addEventListener("drag", (e) => {
			console.log("drag", e);
		});

		this.dispatcher.addEventListener("click", (e) => {
			// console.log("clicked: ", e.hovered.node.name + " - point #" + e.hovered.pointIndex);
		});

		this.material.events.onChange((event) => this.events.dispatcher.dispatch("material_changed", {material: event.material}));
	}

	load(node){

		if(!node.loaded){
			this.loader.loadNode(node);
		}

	}

	updateVisibility(camera, renderer){

		let tStart = performance.now();

		if(this.root?.geometry?.statsList != null){
			this.material.update(this);
		}

		for(let node of this.visibleNodes){
			node.visible = false;
		}

		let visibleNodes = [];
		let loadQueue = [];
		let unfilteredLoadQueue = [];
		let priorityQueue = new BinaryHeap(function (x) { return 1 / x.weight; });

		let camPos = camera.getWorldPosition();
		let world = this.world;
		let view = camera.view;
		let proj = camera.proj;
		_fm.multiplyMatrices(proj, view);
		let frustum = new Frustum();
		frustum.setFromMatrix(_fm);

		priorityQueue.push({ 
			node: this.root, 
			weight: Number.MAX_VALUE,
		});

		let i = 0;
		let numPoints = 0;

		let needsUnfiltered = !["position", "rgba"].includes(Potree.settings.attribute);

		this.root.__pixelSize = 10000;

		while (priorityQueue.size() > 0) {
			let element = priorityQueue.pop();
			let {node, weight} = element;

			if(!node.loaded){

				if(loadQueue.length < 10){
					loadQueue.push(node);
				}

				continue;
			}

			if(needsUnfiltered && !node.unfilteredLoaded){
				if(unfilteredLoadQueue.length < 10){
					unfilteredLoadQueue.push(node);
				}

				continue;
			}

			// let box = node.boundingBox.clone();
			_box.copy(node.boundingBox);
			_box.applyMatrix4(this.world);
			let insideFrustum = frustum.intersectsBox(_box);
			let fitsInPointBudget = numPoints + node.numElements < this.pointBudget;
			let isLowLevel = node.level <= 2;

			let visible = fitsInPointBudget && insideFrustum;
			visible = visible || isLowLevel;

			node.visible = visible;

			if(!visible){
				continue;
			}

			if(Potree.debug.allowedNodes){
				if(Potree.debug.allowedNodes.includes(node.name)){
					visibleNodes.push(node);
					numPoints += node.numElements;
				}
			}else{
				visibleNodes.push(node);
				numPoints += node.numElements;

			}


			for(let child of node.children){
				if(!child){
					continue;
				}

				// let box = child.boundingBox.clone();
				_box.copy(child.boundingBox);
				_box.applyMatrix4(this.world);

				let center = _box.center();
				let radius = _box.min.distanceTo(_box.max) / 2;

				let dx = camPos.x - center.x;
				let dy = camPos.y - center.y;
				let dz = camPos.z - center.z;

				let dd = dx * dx + dy * dy + dz * dz;
				let distance = Math.sqrt(dd);

				let fov = math.toRadians(camera.fov);
				let slope = Math.tan(fov / 2);
				let projFactor = 1 / (slope * distance);

				let weight = radius * projFactor;
				let pixelSize = weight * renderer.getSize().height;

				child.__pixelSize = pixelSize;
				// debugger;

				if(pixelSize < Potree.settings.minNodeSize){
					continue;
				}

				if(distance - radius < 0){
					weight = Number.MAX_VALUE;
				}

				priorityQueue.push({
					node: child, 
					weight: weight
				});
			}

		}

		tStart = performance.now();
		loadQueue.slice(0, 20);
		loadQueue.sort( (a, b) => a.level - b.level);

		unfilteredLoadQueue.slice(0, 20);
		unfilteredLoadQueue.sort( (a, b) => a.level - b.level);

		for(let node of loadQueue){
			this.loader.loadNode(node);
		}

		for(let node of unfilteredLoadQueue){
			this.loader.loadNodeUnfiltered(node);
		}

		this.visibleNodes = visibleNodes;

		let duration = 1000 * (performance.now() - tStart);
		// if(duration > 3){
		// 	console.log(`updateVisibility took long: ${duration} ms`);
		// }

		return duration;
	}

	render(drawstate){
		renderPointsOctree([this], drawstate);
	}

	getBoundingBoxWorld(){

		let bb = this.boundingBox;
		let min = bb.min;
		let max = bb.max;

		let transformed = [
			new Vector3(min.x, min.y, min.z).applyMatrix4(this.world),
			new Vector3(min.x, min.y, max.z).applyMatrix4(this.world),
			new Vector3(min.x, max.y, min.z).applyMatrix4(this.world),
			new Vector3(min.x, max.y, max.z).applyMatrix4(this.world),
			new Vector3(max.x, min.y, min.z).applyMatrix4(this.world),
			new Vector3(max.x, min.y, max.z).applyMatrix4(this.world),
			new Vector3(max.x, max.y, min.z).applyMatrix4(this.world),
			new Vector3(max.x, max.y, max.z).applyMatrix4(this.world),
		];

		let boxWorld = new Box3();

		for(let point of transformed){
			boxWorld.expandByPoint(point);
		}

		return boxWorld;
	}

	parsePoint(node, index){

		let point = {};

		let view = new DataView(node.geometry.buffer);
		let readers = {
			"double"  : view.getFloat64.bind(view),
			"float"   : view.getFloat32.bind(view),
			"int8"    : view.getInt8.bind(view),
			"uint8"   : view.getUint8.bind(view),
			"int16"   : view.getInt16.bind(view),
			"uint16"  : view.getUint16.bind(view),
			"int32"   : view.getInt32.bind(view),
			"uint32"  : view.getUint32.bind(view),
			"int64"   : view.getBigInt64.bind(view),
			"uint64"  : view.getBigUint64.bind(view),
		};

		let attributes = this.attributes;
		let attributeByteOffset = 0;
		for(let i = 0; i < attributes.attributes.length; i++){
			let attribute = attributes.attributes[i];

			let valueOffset = node.numElements * attributeByteOffset + index * attribute.byteSize;

			let value = null;
			let strValue = "";


			if(["XYZ", "position"].includes(attribute.name)){

				let X = view.getFloat32(valueOffset + 0, true);
				let Y = view.getFloat32(valueOffset + 4, true);
				let Z = view.getFloat32(valueOffset + 8, true);

				let shift = this.position;

				value = [X, Y, Z];
				strValue = new Vector3(X, Y, Z).add(shift).toString(1);

			}else if(attribute.numElements === 1){
				let reader = readers[attribute.type.name].bind(view);

				value = reader(valueOffset, true);
				strValue = `${value}`;
			}else{
				
				let reader = readers[attribute.type.name].bind(view);
				let value = [];
				for(let j = 0; j < attribute.numElements; j++){
					
					let _value = reader(valueOffset + j * attribute.type.size, true);
					value.push(_value);
				}

				strValue = "[" + value.map(v => `${v}`).join(", ") + "]";

			}




			let parsedAttribute = {
				name: attribute.name,
				value, strValue,
			};

			point[attribute.name] = parsedAttribute;

			attributeByteOffset += attribute.byteSize;

		}

		return point;

	}

}