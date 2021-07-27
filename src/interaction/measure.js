
import {Potree, Mesh, Vector3, geometries} from "potree";

export class MeasureTool{

	constructor(potree){
		this.potree = potree;
		this.renderer = potree.renderer;
		this.element = potree.renderer.canvas;

		this.mesh = new Mesh("sphere", geometries.sphere);
		this.mesh.scale.set(0.5, 0.5, 0.5);
		this.mesh.renderLayer = 10;
		potree.scene.root.children.push(this.mesh);

		this.element.addEventListener('mousemove', e => {

			let [x, y] = [e.clientX, e.clientY];

			let node = this.mesh;

			Potree.pick(x, y, (result) => {

				if(result.depth !== Infinity){
					node.position.copy(result.position);

					Potree.pickPos = result.position;
				}
			});

		});

		potree.onUpdate( () => {
			let depth = camera.getWorldPosition().distanceTo(this.mesh.position);
			let radius = depth / 50;

			this.mesh.scale.set(radius, radius, radius);

		});

	}

};
