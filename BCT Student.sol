// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
contract StudentData {
 struct Student {
 string name;
 uint256 rollNumber;
 uint256 age;
 }
 Student[] public students;
 receive() external payable {}
 function addStudent(string memory _name, uint256 _rollNumber, uint256 _age) public {
 Student memory newStudent = Student(_name, _rollNumber, _age);
 students.push(newStudent);
 }
 function getStudentCount() public view returns (uint256) {
 return students.length;
 }
}