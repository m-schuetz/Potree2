
import {Geometry, SceneNode, Mesh} from "potree";
import {Box3, Vector3} from "potree";
import {PhongMaterial, ColorMode} from "potree";
import {NormalMaterial, WireframeMaterial} from "potree";

let tmpCanvas = null;
let tmpContext = null;
function getTmpContext(){

	if(tmpCanvas === null){
		tmpCanvas = document.createElement('canvas');
		tmpCanvas.width = 8192;
		tmpCanvas.height = 8192;
		tmpContext = tmpCanvas.getContext('2d');
	}

	return tmpContext;
}

export function load(url, callbacks){

	let workerPath = "./src/misc/GLBLoaderWorker.js";
	let worker = new Worker(workerPath, {type: "module"});

	let root = new SceneNode("glb root");

	worker.onmessage = (e) => {
		
		let geometryData = e.data.geometry;
		let imageBitmap = e.data.imageBitmap;

		let geometry = new Geometry();
		geometry.buffers = geometryData.buffers;
		geometry.indices = geometryData.indices;
		geometry.numElements = geometryData.numElements;
		geometry.boundingBox.min.copy(geometryData.boundingBox.min);
		geometry.boundingBox.max.copy(geometryData.boundingBox.max);

		let mesh = new Mesh("glb mesh", geometry);

		if(imageBitmap){
			mesh.material = new PhongMaterial();
			mesh.material.image = imageBitmap;
			mesh.material.colorMode = ColorMode.TEXTURE;
			mesh.material.imageBuffer = e.data.imageBuffer;
		}else{
			mesh.material = new PhongMaterial();
			mesh.material.image = null;
			mesh.material.colorMode = ColorMode.VERTEX_COLOR;
		}

		root.children.push(mesh);

		if(root.children.length === 1){
			callbacks.onStart(root);
			callbacks.onNode(mesh);
		}else{
			callbacks.onNode(mesh);
		}

	};

	let absoluteUrl = new URL(url, document.baseURI).href;
	worker.postMessage({url: absoluteUrl});


};



