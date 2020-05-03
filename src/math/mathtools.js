
export function toRadians(degrees){
	let radians = Math.PI * (degrees / 180);

	return radians;
}

export function toDegrees(radians){
	let degrees = 180 * radians / Math.PI;

	return degrees;
}