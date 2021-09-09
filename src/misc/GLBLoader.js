
import {Geometry, SceneNode, Mesh} from "potree";
import {Box3, Vector3} from "potree";
import {PhongMaterial, ColorMode} from "potree";
import {NormalMaterial, WireframeMaterial} from "potree";



export function load(url){

	return new Promise(resolve => {

		let workerPath = "./src/misc/GLBLoaderWorker.js";
		let worker = new Worker(workerPath, {type: "module"});

		worker.onmessage = (e) => {
			
			let geometryData = e.data.geometry;
			let imageBitmap = e.data.imageBitmap;

			let geometry = new Geometry();
			geometry.buffers = geometryData.buffers;
			geometry.indices = geometryData.indices;
			geometry.boundingBox.min.copy(geometryData.boundingBox.min);
			geometry.boundingBox.max.copy(geometryData.boundingBox.max);

			let mesh = new Mesh("glb mesh", geometry);

			if(imageBitmap){
				mesh.material = new PhongMaterial();
				mesh.material.image = imageBitmap;
				mesh.material.colorMode = ColorMode.TEXTURE;
				mesh.material.imageBuffer = e.data.imageBuffer;

				const canvas = document.createElement('canvas');
				canvas.width = imageBitmap.width;
				canvas.height = imageBitmap.width;
				const context = canvas.getContext('2d');
				context.drawImage(imageBitmap, 0, 0);
				let imageData = context.getImageData(0, 0, imageBitmap.width, imageBitmap.height);

				mesh.material.imageData = imageData;
			}else{
				mesh.material = new PhongMaterial();
				mesh.material.image = null;
				mesh.material.colorMode = ColorMode.VERTEX_COLOR;
			}

			resolve(mesh);

		};

		let absoluteUrl = new URL(url, document.baseURI).href;
		worker.postMessage({url: absoluteUrl});

	});

};



