
import {toRadians, toDegrees} from "../math/PMath.js";
import {Matrix4, Vector3} from "potree";
import {SceneNode} from "./SceneNode.js";

export class Camera extends SceneNode{

	constructor(name){
		super(name ?? "camera");

		this.fov = 80;
		this.near = 0.01;
		this.far = 10_000;
		this.aspect = 1;
		this.proj = new Matrix4();
		this.view = new Matrix4();

	}

	updateView(){
		this.view.copy(this.world).invert();
	}

	updateProj(){

		let fovy = toRadians(this.fov);
		//let fovy = toRadians(0.5 * this.fov);

		// const near = this.near;
		// let top = near * Math.tan(fovy);
		// let height = 2 * top;
		// let width = this.aspect * height;
		// let left = - 0.5 * width;

		// this.proj.makePerspective( left, left + width, top, top - height, near, this.far);

		this.proj.perspectiveZO(fovy, this.aspect, this.near);

		let remap = new Matrix4();
		remap.elements[10] = -1;
		remap.elements[14] =  1;

		this.proj = remap.multiply(this.proj);
	}

	// u, v in [0, 1]
	// origin: bottom left
	mouseToDirection(u, v){

		let fovRad = toRadians(this.fov);

		let top = Math.tan(fovRad / 2);
		let height = 2 * top;
		let width = this.aspect * height;

		let origin = new Vector3(0, 0, 0).applyMatrix4(this.world);
		
		let dir = new Vector3(
			0.5 * (2.0 * u - 1.0) * width,
			0.5 * (2.0 * v - 1.0) * height,
			-1,
		).applyMatrix4(this.world);

		return dir.sub(origin).normalize();
	}



};