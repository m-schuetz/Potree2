
import { mat4, vec3 } from '../../libs/gl-matrix.js';
import {toRadians, toDegrees} from "../math/PMath.js";

export class Camera{

	constructor(){

		this.fov = 60;
		this.near = 0.1;
		this.far = 1000;
		this.aspect = 1;
		this.proj = mat4.create();
		this.world = mat4.create();
		this.view = mat4.create();

	}

	updateView(){
		mat4.invert(this.view, this.world);
	}

	updateProj(){
		let fovy = toRadians(0.5 * this.fov);

		mat4.perspective(
			this.proj, 
			fovy, 
			this.aspect, 
			this.near, 
			this.far);
	}



};