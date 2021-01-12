
import {toRadians, toDegrees} from "../math/PMath.js";
import {Matrix4} from "../math/math.js";
import {SceneNode} from "./SceneNode.js";

export class Camera extends SceneNode{

	constructor(name){
		super(name ?? "camera");

		this.fov = 80;
		this.near = 0.1;
		this.far = 1000;
		this.aspect = 1;
		this.proj = new Matrix4();
		this.view = new Matrix4();

	}

	updateView(){
		this.view.copy(this.world).invert();
	}

	updateProj(){

		let fovy = toRadians(0.5 * this.fov);

		const near = this.near;
		let top = near * Math.tan(fovy);
		let height = 2 * top;
		let width = this.aspect * height;
		let left = - 0.5 * width;

		this.proj.makePerspective( left, left + width, top, top - height, near, this.far);
	}



};