
import {Potree} from "potree";


export class MeasureTool{

	constructor(renderer){
		this.renderer = renderer;
		this.element = renderer.canvas;

		this.element.addEventListener('mousemove', e => {

			let [x, y] = [e.clientX, e.clientY];

			let node = scene.root.children.find(c => c.constructor.name === "Mesh");

			if(node){
				Potree.pick(x, y, (result) => {

					if(result.depth !== Infinity){
						node.position.copy(result.position);
						node.updateWorld();
					}
				});
			}

		});

	}

};
