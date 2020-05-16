
import { Matrix3 } from './Matrix3.js';
import { Vector3 } from './Vector3.js';

const _vector1 = new Vector3();
const _vector2 = new Vector3();
const _normalMatrix = new Matrix3();

export class Plane{
	constructor(normal, constant){
		this.normal = normal ?? new Vector3(1, 0, 0);
		this.constant = constant ?? 0;
	}

	setComponents(x, y, z, w){
		this.normal.set(x, y, z);
		this.constant = w;

		return this;
	}

	normalize(){
		let inverseNormalLength = 1.0 / this.normal.length();
		this.normal.multiplyScalar(inverseNormalLength);
		this.constant *= inverseNormalLength;

		return this;
	}

	distanceToPoint(point){
		return this.normal.dot( point ) + this.constant;
	}

	applyMatrix4(matrix, optionalNormalMatrix){

		let normalMatrix = optionalNormalMatrix ?? _normalMatrix.getNormalMatrix(matrix);
		let referencePoint = this.coplanarPoint( _vector1 ).applyMatrix4( matrix );
		let normal = this.normal.applyMatrix3( normalMatrix ).normalize();

		this.constant = - referencePoint.dot( normal );

		return this;
	}

	coplanarPoint(target){
		return target.copy( this.normal ).multiplyScalar( -this.constant );
	}
}