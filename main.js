

import {cube, pointCube} from "./src/prototyping/cube.js";

import {Renderer} from "./src/renderer/Renderer.js";
// import {loadTexture, drawTexture} from "./src/prototyping/textures.js";
import {drawRect} from "./src/prototyping/rect.js";

let frame = 0;

async function run(){

	let renderer = new Renderer();

	await renderer.init();

	let node = pointCube;
	let camera = null;

	// let texture = loadTexture("./resources/images/background.jpg");

	let loop = () => {
		frame++;

		let pass = renderer.start();

			renderer.render(pass, node, camera);

			drawRect(renderer, pass, -0.8, -0.8, 0.2, 0.5);

		renderer.finish(pass);

		requestAnimationFrame(loop);
	};
	requestAnimationFrame(loop);

}


run();





