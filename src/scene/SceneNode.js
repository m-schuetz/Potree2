
import {Vector3} from "../math/Vector3.js";
import {Box3} from "../math/Box3.js";
import {Matrix4} from "../math/Matrix4.js";

export class SceneNode{

	constructor(name){
		this.name = name;
		
		this.position = new Vector3(0, 0, 0);
		this.rotation = new Matrix4();
		this.scale = new Vector3(1, 1, 1);
		this.boundingBox = new Box3();

		this.children = [];

		this.world = new Matrix4();
	}

	updateWorld(){

		let {world} = this;

		world.makeIdentity();
		world.scale(this.scale.x, this.scale.y, this.scale.z);
		world.multiplyMatrices(this.rotation, world);
		world.translate(this.position.x, this.position.y, this.position.z);

	}

	getWorldPosition(){
		return new Vector3().applyMatrix4(this.world);
	}

};