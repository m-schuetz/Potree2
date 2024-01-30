
import {Vector3, Vector4, Matrix4, Box3, Sphere, Frustum, math, PMath, EventDispatcher} from "potree";
import {SceneNode, PointCloudOctreeNode, PointCloudMaterial} from "potree";
import {renderPointsOctree} from "potree";
import {BinaryHeap} from "BinaryHeap";

const _box    = new Box3();
const _sphere = new Sphere();
const _fm     = new Matrix4();
const _vec3   = new Vector3();
const _vec4   = new Vector4();

export const REFINEMENT = {
	ADDITIVE:  0,
	REPLACING: 1,
};

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
		this.refinement     = REFINEMENT.ADDITIVE;

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

	updateVisibility_replacing(camera, renderer){
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
		let worldViewProj = new Matrix4().multiply(proj).multiply(view).multiply(world);
		_fm.multiplyMatrices(proj, view);
		let frustum = new Frustum();
		frustum.setFromMatrix(_fm);

		let octreeSize = this.boundingBox.max.x - this.boundingBox.min.x;
		let octreeRadius = Math.sqrt(octreeSize * octreeSize + octreeSize * octreeSize + octreeSize * octreeSize) / 2;
		let framebufferSize = renderer.getSize();

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

				if(loadQueue.length < 40){
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

			// _sphere.fromBox(node.boundingBox);
			_sphere.center.x = 0.5 * (node.boundingBox.min.x + node.boundingBox.max.x);
			_sphere.center.y = 0.5 * (node.boundingBox.min.y + node.boundingBox.max.y);
			_sphere.center.z = 0.5 * (node.boundingBox.min.z + node.boundingBox.max.z);
			_sphere.radius = octreeRadius / (2 ** node.level);
			_sphere.applyMatrix4(this.world);

			let center = _sphere.center;
			let radius = _sphere.radius;

			let dx = camPos.x - center.x;
			let dy = camPos.y - center.y;
			let dz = camPos.z - center.z;

			let dd = dx * dx + dy * dy + dz * dz;
			let distance = Math.sqrt(dd);

			let fov = math.toRadians(camera.fov);
			let slope = Math.tan(fov / 2);
			let projFactor = 1 / (slope * distance);

			let pixelSize = radius * projFactor * framebufferSize.height;
			let insideFrustum = frustum.intersectsSphere(_sphere);
			// let fitsInPointBudget = numPoints + node.numElements < this.pointBudget;
			let isLowLevel = node.level <= 2;

			// if(!fitsInPointBudget) break;

			// let visible = fitsInPointBudget && insideFrustum;
			// visible = visible || isLowLevel;

			// node.visible = visible;

			let visible = insideFrustum;
			let isLeaf = node.children.every(n => n === null);
			let allChildrenLoaded = node.children.every(n => n === null || n.loaded);
			let shouldBreak = !isLeaf && insideFrustum && pixelSize > Potree.settings.minNodeSize * 2;


			if(shouldBreak && !allChildrenLoaded){
				for(let child of node.children){
					if(!child) continue;

					loadQueue.push(child);
				}

				node.visible = true;
				visibleNodes.push(node);
				numPoints += node.numElements;
			}else if(visible && !shouldBreak){
				// highest LOD node we want to draw

				node.visible = true;
				visibleNodes.push(node);
				numPoints += node.numElements;
			}else if(visible && shouldBreak){
				// we want to draw higher LOD nodes
				for(let child of node.children){
					if(!child) continue;

					_sphere.center.x = 0.5 * (node.boundingBox.min.x + node.boundingBox.max.x);
					_sphere.center.y = 0.5 * (node.boundingBox.min.y + node.boundingBox.max.y);
					_sphere.center.z = 0.5 * (node.boundingBox.min.z + node.boundingBox.max.z);
					_sphere.radius = octreeRadius / (2 ** node.level);
					_sphere.applyMatrix4(this.world);

					let center = _sphere.center;
					let radius = _sphere.radius;

					let dx = camPos.x - center.x;
					let dy = camPos.y - center.y;
					let dz = camPos.z - center.z;

					let dd = dx * dx + dy * dy + dz * dz;
					let distance = Math.sqrt(dd);

					let fov = math.toRadians(camera.fov);
					let slope = Math.tan(fov / 2);
					let projFactor = 1 / (slope * distance);

					let weight = radius * projFactor;
					let pixelSize = weight * framebufferSize.height;

					child.__pixelSize = pixelSize;

					if(distance - radius < 0){
						weight = Number.MAX_VALUE;
					}

					_vec4.set(
						0.5 * (node.boundingBox.min.x + node.boundingBox.max.x),
						0.5 * (node.boundingBox.min.y + node.boundingBox.max.y),
						0.5 * (node.boundingBox.min.z + node.boundingBox.max.z),
						1.0
					);
					_vec4.applyMatrix4(worldViewProj);
					
					if(_vec4.w > 0){
						// weight nodes in center of screen higher
						let u = _vec4.x / _vec4.w;
						let v = _vec4.y / _vec4.w;
						u = math.clamp(u, -1, 1);
						v = math.clamp(v, -1, 1);

						let d = Math.sqrt(u * u + v * v);
						let wCenter = math.clamp(1 - d, 0, 1) + 0.5;

						weight = pixelSize * wCenter;
					}else{
						weight = pixelSize;
					}

					child.weight = weight;


					priorityQueue.push({
						node: child, 
						weight: weight
					});
				}
			}else{
				// we don't want to draw this subtree
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

	updateVisibility_additive(camera, renderer){
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
		let worldViewProj = new Matrix4().multiply(proj).multiply(view).multiply(world);
		_fm.multiplyMatrices(proj, view);
		let frustum = new Frustum();
		frustum.setFromMatrix(_fm);

		let octreeSize = this.boundingBox.max.x - this.boundingBox.min.x;
		let octreeRadius = Math.sqrt(octreeSize * octreeSize + octreeSize * octreeSize + octreeSize * octreeSize) / 2;
		let framebufferSize = renderer.getSize();

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

			// _sphere.fromBox(node.boundingBox);
			_sphere.center.x = 0.5 * (node.boundingBox.min.x + node.boundingBox.max.x);
			_sphere.center.y = 0.5 * (node.boundingBox.min.y + node.boundingBox.max.y);
			_sphere.center.z = 0.5 * (node.boundingBox.min.z + node.boundingBox.max.z);
			_sphere.radius = octreeRadius / (2 ** node.level);
			_sphere.applyMatrix4(this.world);

			// _box.copy(node.boundingBox);
			// _box.applyMcatrix4(this.world);
			// let insideFrustum = frustum.intersectsBox(_box);
			let insideFrustum = frustum.intersectsSphere(_sphere);
			// let fitsInPointBudget = numPoints + node.numElements < this.pointBudget;
			let isLowLevel = node.level <= 2;

			// if(!fitsInPointBudget) break;

			// let visible = fitsInPointBudget && insideFrustum;
			let visible = true;
			visible = visible && insideFrustum;
			visible = visible || isLowLevel;

			node.visible = visible;

			if(!visible){
				continue;
			}

			if(Potree.debug.allowedNodes){
				if(Potree.debug.allowedNodes.includes(node.name)){
					node.visible = true;
					visibleNodes.push(node);
					numPoints += node.numElements;
				}
			}else{
				node.visible = true;
				visibleNodes.push(node);
				numPoints += node.numElements;

			}


			for(let child of node.children){
				if(!child){
					continue;
				}

				_sphere.center.x = 0.5 * (node.boundingBox.min.x + node.boundingBox.max.x);
				_sphere.center.y = 0.5 * (node.boundingBox.min.y + node.boundingBox.max.y);
				_sphere.center.z = 0.5 * (node.boundingBox.min.z + node.boundingBox.max.z);
				_sphere.radius = octreeRadius / (2 ** node.level);
				_sphere.applyMatrix4(this.world);

				let center = _sphere.center;
				let radius = _sphere.radius;

				let dx = camPos.x - center.x;
				let dy = camPos.y - center.y;
				let dz = camPos.z - center.z;

				let dd = dx * dx + dy * dy + dz * dz;
				let distance = Math.sqrt(dd);

				let fov = math.toRadians(camera.fov);
				let slope = Math.tan(fov / 2);
				let projFactor = 1 / (slope * distance);

				let weight = radius * projFactor;
				let pixelSize = weight * framebufferSize.height;

				child.__pixelSize = pixelSize;
				// debugger;

				if(pixelSize < Potree.settings.minNodeSize){
					continue;
				}

				if(distance - radius < 0){
					weight = Number.MAX_VALUE;
				}

				_vec4.set(
					0.5 * (node.boundingBox.min.x + node.boundingBox.max.x),
					0.5 * (node.boundingBox.min.y + node.boundingBox.max.y),
					0.5 * (node.boundingBox.min.z + node.boundingBox.max.z),
					1.0
				);
				_vec4.applyMatrix4(worldViewProj);
				
				if(_vec4.w > 0){
					// weight nodes in center of screen higher
					let u = _vec4.x / _vec4.w;
					let v = _vec4.y / _vec4.w;
					u = math.clamp(u, -1, 1);
					v = math.clamp(v, -1, 1);

					let d = Math.sqrt(u * u + v * v);
					let wCenter = math.clamp(1 - d, 0, 1) + 0.5;

					weight = pixelSize * wCenter;
				}else{
					weight = pixelSize;
				}
					// weight = pixelSize;

				child.weight = weight;

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

	updateVisibility(camera, renderer){

		let camPos = camera.getWorldPosition();
		let world = this.world;
		let view = camera.view;
		let proj = camera.proj;
		let worldViewProj = new Matrix4().multiply(proj).multiply(view).multiply(world);
		_fm.multiplyMatrices(proj, view);
		let frustum = new Frustum();
		frustum.setFromMatrix(_fm);

		let octreeSize = this.boundingBox.max.x - this.boundingBox.min.x;
		let octreeRadius = Math.sqrt(octreeSize * octreeSize + octreeSize * octreeSize + octreeSize * octreeSize) / 2;

		if(this.refinement === REFINEMENT.ADDITIVE){
			return this.updateVisibility_additive(camera, renderer);
		}else{

			this.root.traverse(node => {
				node.visible = false;
			});

			let duration = this.updateVisibility_additive(camera, renderer);

			let replacingVisibleNodes = [];
			let needsUnfiltered = !["position", "rgba"].includes(Potree.settings.attribute);

			for(let node of this.visibleNodes){

				let anyChildVisible = false;
				let allChildrenLoaded = true;
				for(let child of node.children){
					if(child == null) continue;

					anyChildVisible = anyChildVisible || child.visible;
					allChildrenLoaded = allChildrenLoaded && child.loaded;
				}

				if(anyChildVisible && !allChildrenLoaded){
					for(let child of node.children){
						if(child == null) continue;

						this.loader.loadNode(child);

						if(needsUnfiltered){
							this.loader.loadNodeUnfiltered(child);
						}
					}
				}else if(anyChildVisible && allChildrenLoaded){

					// make children visible and hide this node
					for(let child of node.children){
						if(child == null) continue;

						_sphere.center.x = 0.5 * (child.boundingBox.min.x + child.boundingBox.max.x);
						_sphere.center.y = 0.5 * (child.boundingBox.min.y + child.boundingBox.max.y);
						_sphere.center.z = 0.5 * (child.boundingBox.min.z + child.boundingBox.max.z);
						_sphere.radius = octreeRadius / (2 ** child.level);
						_sphere.applyMatrix4(this.world);

						let insideFrustum = frustum.intersectsSphere(_sphere);

						if(insideFrustum && !child.visible){
							child.visible = true;
							replacingVisibleNodes.push(child);
						}

						node.visible = false;
					}

				}else{
					replacingVisibleNodes.push(node);
				}

				this.visibleNodes = replacingVisibleNodes;
				// console.log(this.visibleNodes);
			}

			return duration;

			// return this.updateVisibility_replacing(camera, renderer);
		}
		
	}

	render(drawstate){
		renderPointsOctree([this], drawstate);
	}

	traverse(callback){

		callback(this);

		for(let child of this.children){
			if(child == null) continue;

			child.traverse(callback);
		}

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