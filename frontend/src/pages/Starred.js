import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getCurrentUser, getStarredImages, addStarredImage, removeStarredImage, debugDatabase } from '../services/api';

const Starred = () => {
  const [starredImages, setStarredImages] = useState({});
  const [selectedImage, setSelectedImage] = useState(null);
  const [noteText, setNoteText] = useState('');
  const [showNoteModal, setShowNoteModal] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const currentUser = getCurrentUser();
    if (!currentUser) {
      navigate('/login');
      return;
    }
    
    // Load starred images from MongoDB
    const loadStarredImages = async () => {
      try {
        setLoading(true);
        const currentUser = getCurrentUser();
        console.log('Current user:', currentUser);

        // First check database state with debug API
        try {
          const debugData = await debugDatabase(currentUser.id);
          console.log('Debug data:', debugData);
        } catch (debugError) {
          console.error('Debug API error (non-fatal):', debugError);
        }

        const savedImages = await getStarredImages(currentUser.id);
        console.log('API Response for starred images:', savedImages);
        console.log('Loaded starred images:', Object.keys(savedImages).length);
        
        // Check if images have image_data
        Object.entries(savedImages).forEach(([id, image]) => {
          if (!image.image_data) {
            console.warn(`Image ${id} is missing image_data`);
          }
        });
        
        setStarredImages(savedImages);
      } catch (error) {
        console.error('Error loading starred images:', error);
        setError('Failed to load starred images. Please try again later.');
        setStarredImages({});
      } finally {
        setLoading(false);
      }
    };

    loadStarredImages();
  }, [navigate]);

  const handleRemoveStar = async (imageId) => {
    try {
      const currentUser = getCurrentUser();
      if (!currentUser) {
        navigate('/login');
        return;
      }
      
      setIsSaving(true);
      await removeStarredImage(currentUser.id, imageId);
      
      const updatedImages = { ...starredImages };
      delete updatedImages[imageId];
      setStarredImages(updatedImages);
      
      if (selectedImage && selectedImage.image_id === imageId) {
        setSelectedImage(null);
      }
    } catch (error) {
      console.error('Error removing star:', error);
      setError('Failed to remove star. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleEditNote = (image) => {
    setSelectedImage(image);
    setNoteText(image.note || '');
    setShowNoteModal(true);
  };

  const handleNavigate = (path) => {
    navigate(path);
  };

  const handleAddNote = async () => {
    if (isSaving) return;
    
    try {
      setIsSaving(true);
      
      if (noteText.trim()) {
        const currentUser = getCurrentUser();
        if (!currentUser) {
          navigate('/login');
          return;
        }
        
        // Update in MongoDB
        await addStarredImage(currentUser.id, selectedImage.image_id, noteText);
        
        // Update local state
        const updatedImage = {
          ...selectedImage,
          note: noteText,
          timestamp: new Date().toISOString()
        };
        
        const updatedImages = {
          ...starredImages,
          [selectedImage.image_id]: updatedImage
        };
        
        setStarredImages(updatedImages);
      }
      
      setShowNoteModal(false);
      setNoteText('');
      setSelectedImage(null);
    } catch (error) {
      console.error('Error updating starred image:', error);
      setError('Failed to update star note. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleCloseModal = () => {
    setShowNoteModal(false);
    setNoteText('');
    setSelectedImage(null);
  };

  const formatDate = (dateString) => {
    const options = { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  // Note Modal Component
  const NoteModal = () => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={handleCloseModal}>
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-96 max-w-[90vw]" onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Edit Star Note
          </h3>
          <button
            onClick={handleCloseModal}
            className="text-gray-400 hover:text-gray-500 dark:text-gray-500 dark:hover:text-gray-400"
          >
            <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <textarea
          id="note-text"
          className="w-full h-32 p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 dark:focus:ring-primary-400 dark:focus:border-primary-400 resize-none bg-white dark:bg-gray-700 text-gray-900 dark:text-white border-gray-300 dark:border-gray-600 placeholder-gray-400 dark:placeholder-gray-500"
          placeholder="Enter your note here..."
          value={noteText}
          onChange={(e) => {
            const newValue = e.target.value;
            requestAnimationFrame(() => {
              e.target.selectionStart = e.target.value.length;
              e.target.selectionEnd = e.target.value.length;
            });
            setNoteText(newValue);
          }}
          autoFocus
          dir="ltr"
          spellCheck="false"
          style={{ 
            textAlign: 'left',
            caretColor: 'auto'
          }}
          onFocus={(e) => {
            e.target.selectionStart = e.target.value.length;
            e.target.selectionEnd = e.target.value.length;
          }}
        />
        <div className="flex justify-end space-x-2">
          <button
            type="button"
            onClick={handleCloseModal}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600"
            disabled={isSaving}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleAddNote}
            className="px-4 py-2 text-sm font-medium text-white bg-primary-600 dark:bg-primary-500 rounded-md hover:bg-primary-700 dark:hover:bg-primary-600"
            disabled={!noteText.trim() || isSaving}
          >
            {isSaving ? 'Saving...' : 'Update Note'}
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      {showNoteModal && <NoteModal />}
      <div className="max-w-6xl mx-auto">
        <div className="bg-white dark:bg-gray-800 shadow-xl rounded-lg overflow-hidden">
          <div className="px-6 py-8 border-b border-gray-200 dark:border-gray-700">
            <div className="flex justify-between items-center">
              <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-yellow-500 mr-2" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                  </svg>
                  Starred Scans
                </h1>
                <p className="mt-2 text-gray-600 dark:text-gray-400">
                  View your starred brain scans and notes
                </p>
              </div>
              <button
                onClick={() => handleNavigate('/history')}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 dark:bg-primary-500 dark:hover:bg-primary-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 dark:focus:ring-offset-gray-900 dark:focus:ring-primary-400"
              >
                Back to History
              </button>
            </div>
          </div>

          {loading ? (
            <div className="flex justify-center items-center p-12">
              <div className="loader w-12 h-12 border-4 border-gray-200 dark:border-gray-700 rounded-full"></div>
            </div>
          ) : error ? (
            <div className="p-6 text-center">
              <p className="text-red-500 dark:text-red-400">{error}</p>
            </div>
          ) : Object.keys(starredImages).length === 0 ? (
            <div className="p-6 text-center">
              <p className="text-gray-500 dark:text-gray-400 mb-4">No starred scans found.</p>
              <button
                onClick={() => handleNavigate('/history')}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-primary-600 dark:text-primary-400 bg-primary-50 dark:bg-primary-900/20 hover:bg-primary-100 dark:hover:bg-primary-900/30 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 dark:focus:ring-offset-gray-900 dark:focus:ring-primary-400"
              >
                Go to History to Star Scans
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 p-6">
              {Object.entries(starredImages).map(([imageId, image]) => (
                <div key={imageId} className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow duration-300">
                  <div className="relative">
                    <img
                      src={`data:image/jpeg;base64,${image.image_data}`}
                      alt={image.filename}
                      className="w-full h-48 object-cover"
                      onError={(e) => {
                        console.error('Error loading image:', image.image_id);
                        e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2YzZjRmNiIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMjQiIGZpbGw9IiM5Y2EzYWYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5JbWFnZSBub3QgZm91bmQ8L3RleHQ+PC9zdmc+';
                      }}
                    />
                    <div className="absolute top-2 right-2 flex space-x-2">
                      <button
                        onClick={() => handleEditNote(image)}
                        className="p-2 bg-white dark:bg-gray-800 rounded-full shadow-md hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 dark:focus:ring-offset-gray-900 dark:focus:ring-primary-400"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
                        </svg>
                      </button>
                      <button
                        onClick={() => handleRemoveStar(imageId)}
                        className="p-2 bg-white dark:bg-gray-800 rounded-full shadow-md hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 dark:focus:ring-offset-gray-900 dark:focus:ring-primary-400"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-yellow-500 dark:text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <div className="p-4">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">{image.filename}</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                      Starred on {formatDate(image.timestamp)}
                    </p>
                    <div className="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
                      <p className="text-sm text-gray-700 dark:text-gray-300">{image.note}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Starred; 