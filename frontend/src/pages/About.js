import React from 'react';
import { Link } from 'react-router-dom';

const About = () => {
  return (
    <div className="bg-white dark:bg-gray-900">
      <div className="max-w-7xl mx-auto py-16 px-4 sm:px-6 lg:py-20 lg:px-8">
        <div className="lg:grid lg:grid-cols-3 lg:gap-8">
          <div>
            <h2 className="text-3xl font-extrabold text-gray-900 dark:text-white">About BrainScan AI</h2>
            <p className="mt-4 text-lg text-gray-500 dark:text-gray-400">
              Our mission is to leverage advanced AI technology to assist healthcare professionals in early detection and accurate diagnosis of brain tumors.
            </p>
          </div>
          <div className="mt-12 lg:mt-0 lg:col-span-2">
            <div className="prose prose-indigo dark:prose-invert prose-lg text-gray-500 dark:text-gray-400">
              <p>
                BrainScan AI is a cutting-edge platform that combines artificial intelligence with medical expertise to provide fast, reliable brain tumor detection from MRI and CT scan images.
              </p>
              <p>
                Our system uses state-of-the-art deep learning algorithms trained on thousands of medical images to identify potential tumors with high accuracy. The platform is designed to assist medical professionals, not replace them, by providing an additional layer of analysis that can help catch abnormalities early.
              </p>
              <p>
                All uploaded images are first validated using Google's Gemini API to ensure they are appropriate brain scans before being processed by our specialized tumor detection model.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Technology Stack */}
      <div className="bg-gray-50 dark:bg-gray-800 py-16 px-4 sm:px-6 lg:py-20 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-extrabold text-gray-900 dark:text-white text-center mb-12">Our Technology</h2>
          <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
            <div className="bg-white dark:bg-gray-700 overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-primary-500 dark:bg-primary-400 rounded-md p-3">
                    <svg className="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                    </svg>
                  </div>
                  <div className="ml-5">
                    <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white">Deep Learning</h3>
                    <p className="mt-2 text-base text-gray-500 dark:text-gray-400">
                      Powered by convolutional neural networks trained on extensive medical imaging datasets.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-700 overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-primary-500 dark:bg-primary-400 rounded-md p-3">
                    <svg className="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                    </svg>
                  </div>
                  <div className="ml-5">
                    <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white">Gemini API Integration</h3>
                    <p className="mt-2 text-base text-gray-500 dark:text-gray-400">
                      Google's Gemini API validates that uploaded images are appropriate brain scans before processing.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-700 overflow-hidden shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0 bg-primary-500 dark:bg-primary-400 rounded-md p-3">
                    <svg className="h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                  </div>
                  <div className="ml-5">
                    <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white">Modern Tech Stack</h3>
                    <p className="mt-2 text-base text-gray-500 dark:text-gray-400">
                      Built with React, Flask, MongoDB Atlas, and Tailwind CSS for a seamless user experience.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* How It Works */}
      <div className="max-w-7xl mx-auto py-16 px-4 sm:px-6 lg:py-24 lg:px-8">
        <div className="lg:grid lg:grid-cols-3 lg:gap-8">
          <div>
            <h2 className="text-3xl font-extrabold text-gray-900 dark:text-white">How It Works</h2>
            <p className="mt-4 text-lg text-gray-500 dark:text-gray-400">
              Our platform makes brain tumor detection simple, accurate, and accessible.
            </p>
          </div>
          <div className="mt-12 lg:mt-0 lg:col-span-2">
            <div className="space-y-12">
              <div className="flex">
                <div className="flex-shrink-0">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 dark:bg-primary-400 text-white">
                    <span className="text-lg font-bold">1</span>
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white">Upload a Brain Scan</h3>
                  <p className="mt-2 text-base text-gray-500 dark:text-gray-400">
                    Upload an MRI or CT scan image through our secure platform. The image is automatically validated to ensure it's an appropriate brain scan using Gemini AI.
                  </p>
                </div>
              </div>

              <div className="flex">
                <div className="flex-shrink-0">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 dark:bg-primary-400 text-white">
                    <span className="text-lg font-bold">2</span>
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white">AI Analysis</h3>
                  <p className="mt-2 text-base text-gray-500 dark:text-gray-400">
                    Our machine learning model processes the image, detecting potential tumors, classifying their type, and generating a confidence score along with other key metrics.
                  </p>
                </div>
              </div>

              <div className="flex">
                <div className="flex-shrink-0">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 dark:bg-primary-400 text-white">
                    <span className="text-lg font-bold">3</span>
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white">Comprehensive Results</h3>
                  <p className="mt-2 text-base text-gray-500 dark:text-gray-400">
                    Receive detailed analysis results including tumor detection status, tumor type, confidence level, recommended precautions, and potential treatment options.
                  </p>
                </div>
              </div>

              <div className="flex">
                <div className="flex-shrink-0">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-primary-500 dark:bg-primary-400 text-white">
                    <span className="text-lg font-bold">4</span>
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg leading-6 font-medium text-gray-900 dark:text-white">Secure History</h3>
                  <p className="mt-2 text-base text-gray-500 dark:text-gray-400">
                    All your uploaded scans and analysis results are securely stored in your account for future reference and to track changes over time.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-primary-700 dark:bg-primary-800">
        <div className="max-w-2xl mx-auto text-center py-16 px-4 sm:py-20 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-extrabold text-white sm:text-4xl">
            Experience the power of AI in brain tumor detection
          </h2>
          <p className="mt-4 text-lg leading-6 text-indigo-200 dark:text-indigo-300">
            Join healthcare professionals worldwide who trust our platform
          </p>
          <Link
            to="/register"
            className="mt-8 inline-flex items-center px-5 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-primary-600 dark:text-primary-500 bg-white dark:bg-gray-900 hover:bg-indigo-50 dark:hover:bg-gray-800 sm:px-8"
          >
            Get Started Today
          </Link>
        </div>
      </div>
    </div>
  );
};

export default About; 