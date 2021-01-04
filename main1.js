

import {cube, pointCube} from "./src/prototyping/cube.js";

import {Renderer} from "./src/renderer/Renderer.js";
import {loadImage, drawImage, drawTexture} from "./src/prototyping/textures.js";
import {drawRect} from "./src/prototyping/rect.js";
import {drawMesh} from "./src/modules/mesh/drawMesh.js";
import {Geometry} from "./src/core/Geometry.js";
import {Mesh} from "./src/modules/mesh/Mesh.js";
import {Points} from "./src/modules/points/Points.js";
import {drawPoints} from "./src/modules/points/drawPoints.js";
import {drawQuads} from "./src/modules/points/drawQuads.js";
import {renderProgressive} from "./src/modules/progressive/renderProgressive.js";
import {Camera} from "./src/scene/Camera.js";
import {mat4, vec3} from './libs/gl-matrix.js';
import {OrbitControls} from "./src/navigation/OrbitControls.js";
import {SceneNode} from "./src/scene/SceneNode.js";

import {Potree} from "./src/Potree.js";

let frame = 0;
let lastFrameStart = 0;

async function run(){

	let renderer = new Renderer();

	window.renderer = renderer;

	await renderer.init();

	let camera = new Camera();
	let cameraDistance = 5;
	mat4.translate(camera.world, camera.world, vec3.fromValues(0, 0, cameraDistance));
	camera.updateView();

	let controls = new OrbitControls(renderer.canvas);
	controls.radius = 20;
	controls.yaw = Math.PI / 4;
	controls.pitch = Math.PI / 5;

	// let mesh = new Mesh("test", cube.buffers, cube.vertexCount);
	// mesh.scale.set(0.3, 0.3, 0.3);
	// mesh.position.set(0, 0, 0);
	// mesh.updateWorld();

	// let geometry = new Geometry();
	// geometry.buffers = pointCube.buffers;
	// geometry.numElements = pointCube.vertexCount;
	// let points = new Points("test", geometry);
	// points.updateWorld();

	Potree.load("./resources/pointclouds/heidentor/metadata.json").then(pointcloud => {
		pointcloud.updateVisibility(camera);
		pointcloud.position.set(3, -3, -6)
		pointcloud.updateWorld();
		window.pointcloud = pointcloud;
	});
	

	// let renderTarget = null;

	// {
	// 	let size = [800, 600, 1];
	// 	let rt = {};

	// 	{ // color
	// 		let {device} = renderer;

	// 		let descriptor = {
	// 			size: size,
	// 			format: renderer.swapChainFormat,
	// 			usage: GPUTextureUsage.SAMPLED 
	// 				| GPUTextureUsage.COPY_SRC 
	// 				| GPUTextureUsage.COPY_DST 
	// 				| GPUTextureUsage.OUTPUT_ATTACHMENT,
	// 		};

	// 		rt.texture = device.createTexture(descriptor);
	// 	}

	// 	{ // depth
	// 		let descriptor = {
	// 			size: size,
	// 			format: "depth24plus-stencil8",
	// 			usage: GPUTextureUsage.OUTPUT_ATTACHMENT,
	// 		};

	// 		rt.depthTexture = renderer.device.createTexture(descriptor);
	// 	}

	// 	renderTarget = rt;
	// }



	// let image = await loadImage("./resources/images/background.jpg");

	let loop = () => {
		// let timeSinceLastFrame = performance.now() - lastFrameStart;
		// console.log(timeSinceLastFrame.toFixed(1));
		// lastFrameStart = performance.now();
		frame++;

		controls.update();
		mat4.copy(camera.world, controls.world);
		camera.updateView();

		let size = renderer.getSize();
		camera.aspect = size.width / size.height;
		camera.updateProj();

		// { // draw to texture
		// 	let renderPassDescriptor = {
		// 		colorAttachments: [
		// 			{
		// 				attachment: renderTarget.texture.createView(),
		// 				loadValue: { r: 0.3, g: 0.2, b: 0.1, a: 1.0 },
		// 			},
		// 		],
		// 		depthStencilAttachment: {
		// 			attachment: renderTarget.depthTexture.createView(),
		// 			depthLoadValue: 1.0,
		// 			depthStoreOp: "store",
		// 			stencilLoadValue: 0,
		// 			stencilStoreOp: "store",
		// 		},
		// 		sampleCount: 1,
		// 	};

		// 	const commandEncoder = renderer.device.createCommandEncoder();
		// 	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

		// 	let pass = {commandEncoder, passEncoder, renderPassDescriptor};
		// 	drawQuads(renderer, pass, points, camera);

		// 	passEncoder.endPass();

		// 	renderer.device.defaultQueue.submit([commandEncoder.finish()]);
		// }

		
		// let rt = renderProgressive(renderer, points, camera);
		

		{ // draw to window
			let pass = renderer.start();

			// drawQuads(renderer, pass, points, camera);
			// drawMesh(renderer, pass, mesh, camera);
			// drawTexture(renderer, pass, renderTarget.texture, 0.3, 0.3, 0.4, -0.4);
			// drawTexture(renderer, pass, rt.colorAttachments[0].texture, -0.3, -0.3, 0.4, -0.4);

			if(window.pointcloud){
				window.pointcloud.updateVisibility(camera);
				Potree.render(renderer, pass, window.pointcloud, camera);
			}

			// drawImage(renderer, pass, image, -0.9, 0.9, 0.4, -0.4);
			// drawPoints(renderer, pass, points, camera);
			// drawRect(renderer, pass, -0.8, -0.8, 0.2, 0.5);

			renderer.finish(pass);
		}

		// {
		// 	const commandEncoder = renderer.device.createCommandEncoder();
		// 	commandEncoder.copyTextureToTexture(
		// 		{
		// 			texture: renderTarget.texture,
		// 		},{
		// 			texture: renderer.swapChain.getCurrentTexture(),
		// 		},{
		// 			size: {width: 100, height: 100}
		// 		}
		// 	);
		// 	renderer.device.defaultQueue.submit([commandEncoder.finish()]);
		// }

		requestAnimationFrame(loop);
	};
	requestAnimationFrame(loop);

}


run();





