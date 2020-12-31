
import { mat4, vec3 } from '../../libs/gl-matrix.js';

export class OrbitControls{

	constructor(element){

		this.element = element;
		this.radius = 5;
		this.yaw = 0;
		this.pitch = 0;
		this.world = mat4.create();

		element.addEventListener('mousemove', e => {

			let drag = e.buttons > 0 && e.button === 0;

			if(drag){
				let diffX = e.movementX;
				let diffY = e.movementY;

				let ux = diffX / this.element.width;
				let uy = diffY / this.element.height;

				this.yaw += 6 * ux;
				this.pitch += 6 * uy;

			}
		});

		element.addEventListener('wheel', e => {
			let diff = Math.sign(e.deltaY);

			if(diff > 0){
				this.radius *= 1.05;
			}else if(diff < 0){
				this.radius /= 1.05;
			}

		});
	}

	update(delta){
		mat4.identity(this.world);
		mat4.rotate(
			this.world,
			this.world,
			-this.yaw,
			vec3.fromValues(0, 1, 0)
		);
		mat4.rotate(
			this.world,
			this.world,
			-this.pitch,
			vec3.fromValues(1, 0, 0)
		);
		mat4.translate(this.world, this.world, vec3.fromValues(0, 0, this.radius));
	}



};