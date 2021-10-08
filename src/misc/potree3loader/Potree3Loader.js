
import {Box3, Vector3, Mesh, PhongMaterial} from "potree";
import {renderVoxelsLOD} from "./renderVoxelsLOD.js";

let tmpVec3 = new Vector3();
function createChildAABB(aabb, index){
	let min = aabb.min.clone();
	let max = aabb.max.clone();
	let size = tmpVec3.copy(max).sub(min);

	if ((index & 0b0001) > 0) {
		min.z += size.z / 2;
	} else {
		max.z -= size.z / 2;
	}

	if ((index & 0b0010) > 0) {
		min.y += size.y / 2;
	} else {
		max.y -= size.y / 2;
	}
	
	if ((index & 0b0100) > 0) {
		min.x += size.x / 2;
	} else {
		max.x -= size.x / 2;
	}

	return new Box3(min, max);
}

class Node{

	constructor(){
		this.name = "";
		this.boundingBox = new Box3();
		this.voxels = null;
		this.mesh = null;
		this.parent = null;
		this.children = new Array(8).fill(null);
		this.isLeaf = true;
		this.level = 0;
		this.voxelGridSize = 0;
		this.visible = false;
		this.loaded = false;
	}

	traverse(callback){
		callback(this);

		for(let child of this.children){
			if(child){
				child.traverse(callback);
			}
		}
	}

}


export class Potree3Loader{

	constructor(){

	}

	static async load(url){

		let rMetadata = await fetch(`${url}/metadata.json`);
		let jsMetadata = await rMetadata.json();

		let root = null;
		let nodes = [];
		let nodeMap = new Map();

		for(let nodeName of jsMetadata.nodes){
			let node = new Node();
			node.name = nodeName;
			node.level = node.name.length - 1;

			nodes.push(node);
			nodeMap.set(nodeName, node);
		}

		for(let node of nodes){
			
			if(node.name === "r"){
				root = node;
			}else{
				let parentName = node.name.slice(0, -1);
				let parentNode = nodeMap.get(parentName);

				let childIndex = Number(node.name.slice(-1, Infinity));

				parentNode.children[childIndex] = node;
				node.parent = parentNode;
			}
		}

		root.boundingBox = new Box3(
			new Vector3(...jsMetadata.boundingBox.min),
			new Vector3(...jsMetadata.boundingBox.max),
		);

		root.traverse( (node) => {
			
			for(let childIndex = 0; childIndex < 8; childIndex++){
				let child = node.children[childIndex];

				if(child){
					node.isLeaf = false;

					child.boundingBox = createChildAABB(node.boundingBox, childIndex);
				}
			}

			node.voxelGridSize = jsMetadata.gridSize;

		});

		root.traverse( (node) => {
			console.log(node);

			if(!node.isLeaf){

				node.load = async () => {
					// load voxels
					let pPositions = fetch(`${url}/${node.name}_voxels_positions.bin`);
					let pColors = fetch(`${url}/${node.name}_voxels_colors.bin`);

					Promise.all([pPositions, pColors]).then(async result => {
						let [rPositions, rColors] = result;

						let bPositions = await rPositions.arrayBuffer();
						let bColors = await rColors.arrayBuffer();

						let positions = new Float32Array(bPositions);
						let colors = new Uint8Array(bColors);

						let numVoxels = positions.length / 3;

						node.voxels = {positions, colors, numVoxels};

					});

					node.loaded = true;

					node.load = () => {};
				}
				
			}else{
				// load mesh
				
				node.load = async () => {
					let pPositions = fetch(`${url}/${node.name}_mesh_positions.bin`);
					let pUVs = fetch(`${url}/${node.name}_mesh_uvs.bin`);
					let pColors = fetch(`${url}/${node.name}_mesh_colors.bin`);

					Promise.all([pPositions, pUVs, pColors]).then(async result => {
						let [rPositions, rUVs, rColors] = result;

						let bPositions = await rPositions.arrayBuffer();
						let bUVs = await rUVs.arrayBuffer();
						let bColors = await rColors.arrayBuffer();

						let positions = new Float32Array(bPositions);
						let uvs = new Float32Array(bUVs);
						let colors = new Uint8Array(bColors);

						let numVertices = positions.length / 3;

						let geometry = {
							buffers: [
								{name: "position", buffer: positions},
								{name: "uv", buffer: uvs},
								{name: "color", buffer: colors},
							],
							numElements: numVertices,
						};
						let mesh = new Mesh("abc", geometry);
						mesh.material = new PhongMaterial();
						mesh.material.colorMode = 0;

						// node.mesh = {positions, uvs, colors};
						node.mesh = mesh;

						scene.root.children.push(mesh);


					});
					
					node.loaded = true;

					node.load = () => {};
				}


			}
		});

		potree.onUpdate( () => {
			root.traverse( (node) => {

				{

					let center = node.boundingBox.center();
					let size = node.boundingBox.size().length();
					let camWorldPos = camera.getWorldPosition();
					let distance = camWorldPos.distanceTo(center);
					let visible = (size / distance) > 0.3 / Potree.settings.debugU;

					if(visible && !node.loaded){
						node.load();
						node.visible = false;

						return;
					}

					if(node.name === "r"){
						visible = true;
					} 

					node.visible = visible;

					if(node.mesh){
						node.mesh.visible = visible;
					}

					// if(node.visible){
					// 	potree.renderer.drawBoundingBox(
					// 		node.boundingBox.center(),
					// 		node.boundingBox.size(),
					// 		new Vector3(255, 255, 0),
					// 	);
					// }



				}

			});

			
		});

		potree.renderer.onDraw(drawstate => {
			renderVoxelsLOD(root, drawstate);
		});

	}

}