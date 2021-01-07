
import { mat4, vec3 } from '../../libs/gl-matrix.js';
import {Vector3} from "../math/Vector3.js";
import {Matrix4} from "../math/Matrix4.js";

export class OrbitControls{

	constructor(element){

		this.element = element;
		this.radius = 5;
		this.yaw = 0;
		this.pitch = 0;
		this.pivot = new Vector3();
		this.world = new Matrix4();

		// this.world = mat4.create();


		element.addEventListener('mousemove', e => {

			let dragLeft = e.buttons === 1;
			let dragRight = e.buttons === 2;

			if(dragLeft){
				let diffX = e.movementX;
				let diffY = e.movementY;

				let ux = diffX / this.element.width;
				let uy = diffY / this.element.height;

				this.yaw += 6 * ux;
				this.pitch += 6 * uy;
			}else if(dragRight){
				let diffX = e.movementX;
				let diffY = e.movementY;

				let ux = diffX / this.element.width;
				let uy = diffY / this.element.height;


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

	getPosition(){
		return new Vector3().applyMatrix4(this.world);
	}

	update(delta){
		// mat4.identity(this.world);
		// mat4.translate(this.world, this.world, vec3.fromValues(this.pivot.x, this.pivot.z, -this.pivot.y));
		// mat4.rotate(
		// 	this.world, this.world, -this.yaw,
		// 	vec3.fromValues(0, 1, 0)
		// );
		// mat4.rotate(
		// 	this.world,
		// 	this.world,
		// 	-this.pitch,
		// 	vec3.fromValues(1, 0, 0)
		// );
		// mat4.translate(this.world, this.world, vec3.fromValues(0, 0, this.radius));

		let flip = new Matrix4().set(
			1, 0, 0, 0,
			0, 0, 1, 0,
			0, -1, 0, 0,
			0, 0, 0, 1,
		);

		this.world.makeIdentity();
		this.world.translate(0, 0, this.radius);
		this.world.multiplyMatrices(flip, this.world);
		this.world.rotate(Math.PI / 2 - this.pitch, new Vector3(1, 0, 0));
		this.world.rotate(-this.yaw, new Vector3(0, 1, 0));

		this.world.translate(this.pivot.x, this.pivot.z, -this.pivot.y);
	}



};